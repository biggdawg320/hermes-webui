"""Regression coverage for #8106 — docker_init.bash first-start home-ownership
race with concurrent writers into the shared ~/.hermes.

Background: on a fresh container, ``chown_home_hermeswebui`` walks
``/home/hermeswebui`` with ``find ... -exec chown {} +``. If the agent
container is concurrently writing to the shared home (for example the
``os.replace``-style atomic rename ``gateway/status.py`` uses for
``gateway-starts.log``), a filename ``find`` discovered via ``readdir`` can be
renamed away before ``chown`` is invoked on it. ``chown`` then reports that
one name as "No such file or directory", and because ``find`` returns
non-zero whenever any invocation of the exec'd command fails, the whole walk
aborts under ``set -e`` (surfaced at the call site as
``error_exit "Failed to set owner of /home/hermeswebui"``).

This reproduces that race directly against the real, unmodified
``find``/``chown`` walk extracted from ``docker_init.bash``, using the test
process's own uid/gid so a failure can only come from the vanished-file race,
never from a permission error.
"""
from pathlib import Path
import os
import shutil
import subprocess

import pytest


REPO = Path(__file__).resolve().parents[1]
INIT_SCRIPT = (REPO / "docker_init.bash").read_text(encoding="utf-8")

MAX_ATTEMPTS = 40
FILE_COUNT = 8000
RENAMER_COUNT = 8


def _extract_chown_home_hermeswebui() -> str:
    start = INIT_SCRIPT.index("chown_home_hermeswebui()")
    end = INIT_SCRIPT.index("\n}\n", start) + len("\n}\n")
    return INIT_SCRIPT[start:end]


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_chown_home_hermeswebui_tolerates_concurrent_atomic_rename(tmp_path):
    home = tmp_path / "home" / "hermeswebui"
    home.mkdir(parents=True)

    # A tree wide enough that the find+chown walk takes long enough for tight
    # concurrent rename loops to land in the gap between find's directory scan
    # and chown's pass over the same names — mirroring the issue's own
    # reproduction (20,000 files + a background rename loop: 4/10 runs failed,
    # 0/10 without the background loop).
    for i in range(FILE_COUNT):
        (home / f"f{i}").write_bytes(b"")

    fn_src = _extract_chown_home_hermeswebui().replace(
        "/home/hermeswebui", str(home)
    )

    uid = os.getuid()
    gid = os.getgid()

    script = f"""
set -e
WANTED_UID={uid}
WANTED_GID={gid}
{fn_src}
chown_home_hermeswebui
"""

    renamers = [
        subprocess.Popen(
            [
                "bash",
                "-c",
                f'while true; do : > "{home}/r{i}.tmp"; '
                f'mv -f "{home}/r{i}.tmp" "{home}/r{i}"; done',
            ]
        )
        for i in range(RENAMER_COUNT)
    ]

    try:
        failures = []
        for attempt in range(1, MAX_ATTEMPTS + 1):
            result = subprocess.run(
                ["bash", "-c", script],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode != 0:
                failures.append((attempt, result.stdout, result.stderr))
                break
    finally:
        for p in renamers:
            p.terminate()
        for p in renamers:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()

    assert not failures, (
        "chown_home_hermeswebui (docker_init.bash) aborted the home-ownership "
        f"walk on attempt {failures[0][0]}/{MAX_ATTEMPTS} because a file "
        "vanished mid-walk from a concurrent atomic rename, exactly the first-"
        "start race in #8106 where the agent's os.replace-style writes into "
        "the shared ~/.hermes make `find -exec chown {} +` hit ENOENT on one "
        f"name and abort the whole walk. stderr: {failures[0][2].strip()!r}"
    )
