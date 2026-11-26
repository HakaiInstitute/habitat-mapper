"""Check GitHub Releases for a newer version of the desktop app."""

from __future__ import annotations

import platform
from dataclasses import dataclass

import requests
from loguru import logger
from packaging.version import InvalidVersion, Version

RELEASES_URL = "https://api.github.com/repos/HakaiInstitute/habitat-mapper/releases/latest"

# End of the installer file names made by .github/workflows/desktop-build.yml, by platform.system()
_INSTALLER_SUFFIXES = {"Windows": "-windows-x64-setup.exe", "Darwin": "-macos-arm64.dmg"}


@dataclass(frozen=True)
class Release:
    """A release that is newer than the running app."""

    version: str
    page_url: str
    """The release page on GitHub, with the release notes and every download."""
    installer_url: str | None
    """The installer for this platform, or None if the release doesn't have one."""

    @property
    def download_url(self) -> str:
        """The installer if there is one for this platform, otherwise the release page."""
        return self.installer_url or self.page_url


def newer_release(current_version: str, system: str | None = None, timeout: float = 5) -> Release | None:
    """Find out whether a newer release than the running version is available.

    Network and parsing errors are logged at debug level and treated as no update, since the check isn't essential.

    Args:
        current_version: Version of the running app, e.g. "0.17.0".
        system: Platform to find the installer for, as returned by platform.system(). Defaults to this one.
        timeout: Request timeout in seconds.

    Returns:
        The newer release, or None.
    """
    try:
        response = requests.get(RELEASES_URL, timeout=timeout, headers={"Accept": "application/vnd.github+json"})
        response.raise_for_status()
        release = response.json()
        latest = release["tag_name"].removeprefix("v")
        if Version(latest) <= Version(current_version):
            return None
        suffix = _INSTALLER_SUFFIXES.get(system or platform.system())
        installer_url = next(
            (
                asset["browser_download_url"]
                for asset in release.get("assets", [])
                if suffix is not None and asset["name"].endswith(suffix)
            ),
            None,
        )
        return Release(latest, release["html_url"], installer_url)
    except (requests.RequestException, KeyError, TypeError, ValueError, InvalidVersion) as e:
        logger.debug(f"Could not check for updates: {e}")
    return None
