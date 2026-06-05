"""Here we put all the constants that we use across our module.

Some examples are paths to data or to figures folders.

.. note::
    If you notice that you often change any of the constants, than it is not a
    constant and it does not belong here.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PATHS:
    """Constant paths"""

    root = Path(".")

    results = root / "results"

    data = root / "data"

    figures = results / "figures"


PATHS.data.mkdir(exist_ok=True)
PATHS.results.mkdir(exist_ok=True)
PATHS.figures.mkdir(exist_ok=True)
