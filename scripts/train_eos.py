"""Train the 2,048-context EOS-learning model from scratch."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from train import main as train_main

DEFAULT_ARGUMENTS = (
    "--config",
    "configs/training_eos_2048.toml",
)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    return train_main([*DEFAULT_ARGUMENTS, *arguments])


if __name__ == "__main__":
    raise SystemExit(main())
