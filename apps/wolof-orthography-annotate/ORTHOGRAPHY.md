# Wolof dual orthography — why the dataset is “noisy”

Wolof is primarily an **oral** language. Written Wolof on the web, in SMS, and in
many scraped corpora mixes (at least) two Latin writing practices. That is not a
bug in a single pipeline step — it is a property of the language community — but
it **is** a serious issue for LLM training if left unlabeled.

## 1. The two poles (simplified)

| Label (this app) | What volunteers should mark | Typical cues |
|---|---|---|
| **standard** | Official / school / CLAD-style Latin Wolof with proper special letters | Diacritics and digraphs: `à é ë ó ñ ŋ`, careful `x`/`q`, consistent word boundaries |
| **non_standard** | Informal “phone / social” spelling | Missing diacritics (`jaap` vs `jàpp`), phonetic swaps (`ñ`→`gn`, `x`→`kh`), fused words, heavy FR code-switch typed as French |

Real text often sits in between → use **mixed**. Non-Wolof / garbage → **not_wolof**.

## 2. Evidence in GalsenAI data

From Hub cards and prior audits (`Experiments/LFM2.5-Base-1B/docs/DATASETS.md`):

1. **`galsenai/wolof_centalized_corpus`** explicitly lists as a known limit:
   *“Orthographe hétérogène : standard et non-standard mélangés (le wolof n'a pas d'orthographe unique standardisée).”*
2. **`soynade-research/Wolof-Non-Standard-Orthography`** (parallel `wo` /
   `non_standardized`, ~3.4k synthetic pairs from real informal patterns) is
   already mixed into the centralized corpus. Documented transforms:
   diacritic stripping, `ñ→gn`, `x→kh`, word merging, French code-switching.
3. **`galsenai/wolof_sft`** includes an `instruction` subtask for **diacritic
   restoration** (`Japp` → `Jàpp`), which only makes sense if both forms appear
   in the wild.
4. **`galsenai/cleaned_data_for_clm`** (~959k train rows, `text` column) is the
   default CPT mix. Exact-dedup + NFC **does not** collapse orthographic
   variants (`jàpp` ≠ `japp`). Card example already looks mixed/noisy:
   `déccefu njóogu 1549/ 6i fan??` — so CPT sees both conventions as different
   strings, which:
   - inflates vocabulary pressure / fertility,
   - splits probability mass across spellings of the same word,
   - confuses eval if train and test prefer different conventions.

## 3. Why this hurts model quality

| Stage | Failure mode |
|---|---|
| Tokenizer extension | Whole-word candidates may favor one spelling; the other stays fragmented |
| CPT | Same lemma competes as two surface forms → higher BPB than a normalized corpus |
| SFT | Model may “correct” or “corrupt” diacritics inconsistently |
| Eval / verbalizer | Surface mismatch looks like modeling error |

## 4. What crowdsourcing fixes

We do **not** ask volunteers to rewrite texts in this v1 app. We ask them to
**label** each existing string so we can later:

- train / evaluate **separately** on standard vs non-standard,
- build normalization pairs,
- rebalance CPT mixtures,
- report honest metrics per orthography.

Consensus (2–3 agreeing volunteers, configurable) reduces single-annotator noise.

## 5. Guidance for annotators (short)

1. Look at **special characters**: `ë ñ ŋ à é ó` → lean **standard**.
2. Look at **phone spellings**: `gn` for ñ, `kh` for x, no accents on clearly
   Wolof words → lean **non_standard**.
3. One sentence with both systems, or unclear → **mixed**.
4. French-only / English-only / code dump → **not_wolof**.
5. When unsure between standard and non_standard, prefer **mixed** over guessing.

## 6. References

- Dataset card: [soynade-research/Wolof-Non-Standard-Orthography](https://huggingface.co/datasets/soynade-research/Wolof-Non-Standard-Orthography)
- Centralized corpus limits: [galsenai/wolof_centalized_corpus](https://huggingface.co/datasets/galsenai/wolof_centalized_corpus)
- ISO / CLAD Latin orthography tradition for Wolof (school materials in Senegal)
