"""Make ``pytest`` work from a fresh checkout without installing the package.

tests/ lives next to src/ in this repo, so an unconditional path insert keeps
plain ``pytest`` green whether or not zhsteer is installed in the environment.
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
