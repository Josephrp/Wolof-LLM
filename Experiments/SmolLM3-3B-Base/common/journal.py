"""Ecriture du journal de run et des courbes."""

import json
from pathlib import Path


def extraire_courbes(trainer):
    hist = trainer.state.log_history
    train = [(h["step"], h["loss"]) for h in hist if "loss" in h]
    val = [(h["step"], h["eval_loss"]) for h in hist if "eval_loss" in h]
    lr = [(h["step"], h["learning_rate"]) for h in hist if "learning_rate" in h]
    return train, val, lr


def tracer(train, val, lr, titre, sortie):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    if train:
        s, l = zip(*train)
        ax[0].plot(s, l, marker="o", ms=4, color="#1f77b4", label="train")
    if val:
        s, l = zip(*val)
        ax[0].plot(s, l, marker="s", ms=5, color="#d62728", lw=2, label="validation")
    ax[0].set_xlabel("pas")
    ax[0].set_ylabel("loss")
    ax[0].set_title(f"{titre} - loss")
    ax[0].legend()
    ax[0].grid(alpha=.3)

    if lr:
        s, v = zip(*lr)
        ax[1].plot(s, v, color="darkorange", lw=2)
    ax[1].set_xlabel("pas")
    ax[1].set_ylabel("learning rate")
    ax[1].set_title("scheduler")
    ax[1].grid(alpha=.3)

    plt.tight_layout()
    plt.savefig(sortie, dpi=200, bbox_inches="tight")
    plt.close()


def ecrire(chemin, contenu):
    Path(chemin).parent.mkdir(parents=True, exist_ok=True)
    with open(chemin, "w", encoding="utf-8") as f:
        json.dump(contenu, f, indent=2, ensure_ascii=False)
