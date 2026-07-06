# SPDX-License-Identifier: MIT

import pytest

@pytest.mark.dashboard
def test_dashboard_placeholder():
    """
    Placeholder test for the dashboard marker.
    This prevents pytest from exiting with code 5 (no tests collected)
    when running `pytest -m dashboard` in environments where dashboard
    integration tests are not present (e.g., open source release).
    """
    pass
