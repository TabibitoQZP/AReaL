# SPDX-License-Identifier: Apache-2.0
"""Import the standalone scripts without making an `agentick` Python package."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
