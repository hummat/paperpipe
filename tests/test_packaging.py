from __future__ import annotations

from pathlib import Path

import pytest

try:
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[import-not-found]


@pytest.mark.parametrize("extra", ["paperqa", "mcp"])
def test_paperqa_extras_constrain_fhlmi_to_router_compatible_versions(extra: str) -> None:
    pyproject = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())

    assert "fhlmi>=0.45,<0.47; python_version >= '3.11'" in pyproject["project"]["optional-dependencies"][extra]
