"""Integration test config: skip if pylibheom is not built."""

import pytest


def _pylibheom_available():
    try:
        import pyheom.pylibheom  # noqa: F401
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _pylibheom_available(),
    reason="pylibheom extension is not built",
)
