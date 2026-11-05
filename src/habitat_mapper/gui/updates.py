"""Check GitHub Releases for a newer version of the desktop app."""

from __future__ import annotations

import requests
from loguru import logger
from packaging.version import InvalidVersion, Version

RELEASES_URL = "https://api.github.com/repos/HakaiInstitute/habitat-mapper/releases/latest"


def newer_release(current_version: str, timeout: float = 5) -> tuple[str, str] | None:
    """Find out whether a newer release than the running version is available.

    Network and parsing errors are logged at debug level and treated as no update, since the check isn't essential.

    Args:
        current_version: Version of the running app, e.g. "0.17.0".
        timeout: Request timeout in seconds.

    Returns:
        The newer version and the URL of its release page, or None.
    """
    try:
        response = requests.get(RELEASES_URL, timeout=timeout, headers={"Accept": "application/vnd.github+json"})
        response.raise_for_status()
        release = response.json()
        latest = release["tag_name"].removeprefix("v")
        if Version(latest) > Version(current_version):
            return latest, release["html_url"]
    except (requests.RequestException, KeyError, ValueError, InvalidVersion) as e:
        logger.debug(f"Could not check for updates: {e}")
    return None
