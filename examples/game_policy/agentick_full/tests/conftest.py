# SPDX-License-Identifier: Apache-2.0
"""Run from this standalone directory without importing AReaL's test fixtures."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
