"""Durable, fail-fast stage execution for the production LFM2.5 pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

from common.tracking import TrackingRun


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def atomic_json_write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    for attempt in range(5):
        try:
            os.replace(temporary, path)
            break
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.05 * (attempt + 1))


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def command_hash(command: list[str], env_names: list[str]) -> str:
    environment = {
        name: hashlib.sha256(os.environ.get(name, "").encode()).hexdigest()
        for name in sorted(env_names)
    }
    return canonical_hash({"command": command, "environment_hashes": environment})


def pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


class PipelineLock(AbstractContextManager["PipelineLock"]):
    """Exclusive PID lock with safe stale-lock recovery."""

    def __init__(self, path: Path):
        self.path = path
        self.acquired = False

    def __enter__(self) -> Self:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {"pid": os.getpid(), "created_at": utc_now()}
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            try:
                current = json.loads(self.path.read_text(encoding="utf-8"))
                existing_pid = int(current["pid"])
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"invalid pipeline lock: {self.path}") from exc
            if pid_is_alive(existing_pid):
                raise RuntimeError(f"pipeline already running with pid {existing_pid}")
            stale = self.path.with_suffix(f".stale.{int(time.time())}")
            os.replace(self.path, stale)
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(record, handle)
        self.acquired = True
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.acquired:
            self.path.unlink(missing_ok=True)
        self.acquired = False


@dataclass(frozen=True)
class ResourceState:
    free_disk_gb: float
    gpu_free_mib: list[int]
    gpu_processes: list[dict[str, str]]


def inspect_resources(root: Path) -> ResourceState:
    free_disk_gb = shutil.disk_usage(root).free / (1024**3)
    gpu_free: list[int] = []
    gpu_processes: list[dict[str, str]] = []
    if shutil.which("nvidia-smi"):
        memory = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        gpu_free = [int(line.strip()) for line in memory.stdout.splitlines() if line.strip()]
        processes = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        for line in processes.stdout.splitlines():
            parts = [part.strip() for part in line.split(",", 2)]
            if len(parts) == 3:
                gpu_processes.append(
                    {"pid": parts[0], "name": parts[1], "used_memory_mib": parts[2]}
                )
    return ResourceState(free_disk_gb, gpu_free, gpu_processes)


def validate_resources(
    resources: ResourceState,
    *,
    min_disk_gb: float,
    min_gpu_free_mib: int,
    require_gpu: bool,
) -> None:
    if resources.free_disk_gb < min_disk_gb:
        raise RuntimeError(
            f"insufficient disk: {resources.free_disk_gb:.1f} GiB free; "
            f"{min_disk_gb:.1f} GiB required"
        )
    if require_gpu and not resources.gpu_free_mib:
        raise RuntimeError("GPU required but nvidia-smi returned no devices")
    if require_gpu and max(resources.gpu_free_mib) < min_gpu_free_mib:
        raise RuntimeError(
            f"insufficient free VRAM: maximum {max(resources.gpu_free_mib)} MiB; "
            f"{min_gpu_free_mib} MiB required"
        )


def outputs_complete(root: Path, outputs: list[str]) -> bool:
    for output in outputs:
        path = root / output
        if not path.exists():
            return False
        if path.is_file() and path.stat().st_size == 0:
            return False
        if path.is_dir() and not any(path.iterdir()):
            return False
    return True


class PipelineRunner:
    def __init__(
        self,
        *,
        root: Path,
        manifest_path: Path,
        state_path: Path,
        logs_dir: Path,
        resume: bool,
        dry_run: bool,
    ):
        self.root = root
        self.manifest_path = manifest_path
        self.state_path = state_path
        self.logs_dir = logs_dir
        self.resume = resume
        self.dry_run = dry_run
        self.manifest = self._load_manifest()
        self.state = self._load_state()

    def _load_manifest(self) -> dict[str, Any]:
        import yaml

        manifest = yaml.safe_load(self.manifest_path.read_text(encoding="utf-8")) or {}
        stages = manifest.get("stages")
        if not isinstance(stages, list) or not stages:
            raise ValueError("pipeline manifest requires a non-empty stages list")
        names = [stage.get("name") for stage in stages]
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("every stage requires a name")
        if len(names) != len(set(names)):
            raise ValueError("pipeline stage names must be unique")
        known: set[str] = set()
        for stage in stages:
            dependencies = stage.get("requires", [])
            if not isinstance(dependencies, list) or not set(dependencies) <= known:
                raise ValueError(f"{stage['name']}: requires must reference earlier stages")
            command = stage.get("command")
            if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
                raise ValueError(f"{stage['name']}: command must be a string list")
            known.add(stage["name"])
        return manifest

    def _load_state(self) -> dict[str, Any]:
        if self.state_path.exists():
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if not self.resume:
                raise RuntimeError(
                    f"state already exists at {self.state_path}; use --resume or a new run id"
                )
            return state
        state = {
            "schema_version": 1,
            "run_id": self.manifest.get("run_id"),
            "manifest": str(self.manifest_path),
            "manifest_sha256": hashlib.sha256(self.manifest_path.read_bytes()).hexdigest(),
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "status": "pending",
            "stages": {},
        }
        atomic_json_write(self.state_path, state)
        return state

    def selected_stages(self, only_size: str | None) -> list[dict[str, Any]]:
        stages = self.manifest["stages"]
        if only_size is None:
            return stages
        selected_names = {
            stage["name"]
            for stage in stages
            if stage.get("size") in {None, "common", only_size}
        }
        changed = True
        while changed:
            changed = False
            for stage in stages:
                if stage["name"] in selected_names:
                    for dependency in stage.get("requires", []):
                        if dependency not in selected_names:
                            selected_names.add(dependency)
                            changed = True
        return [stage for stage in stages if stage["name"] in selected_names]

    def _persist(self) -> None:
        self.state["updated_at"] = utc_now()
        atomic_json_write(self.state_path, self.state)

    def run(self, only_size: str | None = None) -> int:
        stages = self.selected_stages(only_size)
        self.state["status"] = "dry_run" if self.dry_run else "running"
        self.state["pid"] = os.getpid()
        self._persist()
        for stage in stages:
            self._run_stage(stage)
        self.state["status"] = "dry_run_complete" if self.dry_run else "complete"
        self.state["completed_at"] = utc_now()
        self._persist()
        return 0

    def _run_stage(self, stage: dict[str, Any]) -> None:
        name = stage["name"]
        command = stage["command"]
        stage_hash = command_hash(command, stage.get("hash_env", []))
        previous = self.state["stages"].get(name, {})
        outputs = stage.get("outputs", [])
        if (
            self.resume
            and previous.get("status") == "complete"
            and previous.get("command_env_sha256") == stage_hash
            and outputs_complete(self.root, outputs)
        ):
            export_path = stage.get("export_env_json")
            if export_path:
                exported = json.loads((self.root / export_path).read_text(encoding="utf-8"))
                if set(exported) - {
                    "TRACKIO_SPACE_ID",
                    "TRACKIO_BUCKET_ID",
                    "TRACKIO_DATASET_ID",
                }:
                    raise RuntimeError(f"{name}: attempted to export a forbidden environment key")
                os.environ.update({key: str(value) for key, value in exported.items()})
            tracking_metadata_path = stage.get("tracking_metadata_json")
            if tracking_metadata_path:
                previous["tracking"] = json.loads(
                    (self.root / tracking_metadata_path).read_text(encoding="utf-8")
                )
                self.state["stages"][name] = previous
                self._persist()
            return
        for dependency in stage.get("requires", []):
            if self.state["stages"].get(dependency, {}).get("status") != "complete":
                raise RuntimeError(f"{name}: dependency gate {dependency} is not complete")

        resources = inspect_resources(self.root)
        resource_config = stage.get("resources", {})
        if not self.dry_run:
            validate_resources(
                resources,
                min_disk_gb=float(resource_config.get("min_disk_gb", 200)),
                min_gpu_free_mib=int(resource_config.get("min_gpu_free_mib", 0)),
                require_gpu=bool(resource_config.get("require_gpu", False)),
            )
        record = {
            "status": "dry_run" if self.dry_run else "running",
            "started_at": utc_now(),
            "command_env_sha256": stage_hash,
            "command": command,
            "resources_before": {
                "free_disk_gb": round(resources.free_disk_gb, 2),
                "gpu_free_mib": resources.gpu_free_mib,
                "gpu_processes": resources.gpu_processes,
            },
            "attempt": int(previous.get("attempt", 0)) + 1,
        }
        self.state["stages"][name] = record
        self._persist()
        if self.dry_run:
            record["status"] = "complete"
            record["completed_at"] = utc_now()
            self._persist()
            return

        self.logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.logs_dir / f"{name}.log"
        record["log"] = str(log_path)
        tracking_run = None
        if stage.get("tracking", True):
            tracking_run = TrackingRun(
                size=str(stage.get("size", "common")),
                stage=name,
                tokenizer_family=str(stage.get("tokenizer_family", "none")),
                method=str(stage.get("method", "orchestrator")),
                seed=int(stage.get("seed", 42)),
                run_id=str(self.manifest.get("run_id", "production")),
                config={
                    "stage": name,
                    "command_env_sha256": stage_hash,
                    "outputs": outputs,
                    "resources": resource_config,
                },
                production=True,
                state_path=self.state_path.parent / "tracking" / f"{name}.json",
            )
            tracking_run.start()
            record["tracking"] = tracking_run.metadata()
            self._persist()
        try:
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n[{utc_now()}] stage={name} attempt={record['attempt']}\n")
                log.flush()
                result = subprocess.run(
                    command,
                    cwd=self.root,
                    env={
                        **os.environ,
                        "LFM_PRODUCTION": "1",
                        "LFM_PIPELINE_RUN_ID": str(
                            self.manifest.get("run_id", "production")
                        ),
                    },
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            if result.returncode != 0:
                raise RuntimeError(f"{name}: command failed with exit code {result.returncode}")
            if not outputs_complete(self.root, outputs):
                raise RuntimeError(f"{name}: expected outputs are missing or empty")
            export_path = stage.get("export_env_json")
            if export_path:
                exported = json.loads((self.root / export_path).read_text(encoding="utf-8"))
                if set(exported) - {
                    "TRACKIO_SPACE_ID",
                    "TRACKIO_BUCKET_ID",
                    "TRACKIO_DATASET_ID",
                }:
                    raise RuntimeError(f"{name}: attempted to export a forbidden environment key")
                os.environ.update({key: str(value) for key, value in exported.items()})
            tracking_metadata_path = stage.get("tracking_metadata_json")
            if tracking_metadata_path:
                record["tracking"] = json.loads(
                    (self.root / tracking_metadata_path).read_text(encoding="utf-8")
                )
            if tracking_run:
                tracking_run.log(
                    {
                        "gate_status": 1,
                        "free_disk_gb": resources.free_disk_gb,
                        "gpu_free_mib": max(resources.gpu_free_mib or [0]),
                    }
                )
        except BaseException as exc:
            if tracking_run:
                tracking_run.alert("Red pipeline gate", str(exc), level="error")
                tracking_run.finish(success=False, error=exc)
                record["tracking"] = tracking_run.metadata()
            record["status"] = "failed"
            record["failed_at"] = utc_now()
            record["error"] = str(exc)
            self.state["status"] = "failed"
            self._persist()
            raise
        if tracking_run:
            tracking_run.finish(success=True)
            record["tracking"] = tracking_run.metadata()
        record["status"] = "complete"
        record["completed_at"] = utc_now()
        self._persist()
