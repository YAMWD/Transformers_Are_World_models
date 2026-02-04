from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-paper",
        action="store_true",
        default=False,
        help="Run paper-level performance tests (very slow; requires trained checkpoints).",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    run_paper = bool(config.getoption("--run-paper")) or os.environ.get("RUN_PAPER_TESTS") == "1"
    if run_paper:
        return

    skip_paper = pytest.mark.skip(reason="paper-level test (enable with --run-paper or RUN_PAPER_TESTS=1)")
    for item in items:
        if "paper" in item.keywords:
            item.add_marker(skip_paper)
