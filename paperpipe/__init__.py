"""paperpipe package (compatibility exports)."""

from __future__ import annotations

from .cli import cli
from .config import *  # noqa: F403
from .core import *  # noqa: F403
from .install import *  # noqa: F403
from .leann import *  # noqa: F403
from .output import *  # noqa: F403
from .paper import *  # noqa: F403
from .paperqa import *  # noqa: F403
from .search import *  # noqa: F403

__all__ = ["cli"]
