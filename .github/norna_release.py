"""Build this fork and publish it as a GitHub Release tarball (norna-game's playcanvas dependency).

Run by .github/workflows/norna-build.yml on a self-hosted runner, or by hand:
    python3 .github/norna_release.py [--dry-run]

The release tag is v<package.json version>. A tag that already exists is left alone, so a push that
does not bump the version (bump the -norna.N suffix when you add a patch) builds nothing twice.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*cmd: str) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True, text=True)


def out(*cmd: str) -> str:
    return subprocess.run(cmd, cwd=ROOT, check=True, text=True, capture_output=True).stdout.strip()


def main() -> int:
    dry = "--dry-run" in sys.argv
    version = json.loads((ROOT / "package.json").read_text())["version"]
    tag = f"v{version}"
    if subprocess.run(["gh", "release", "view", tag], cwd=ROOT, capture_output=True).returncode == 0:
        print(f"release {tag} already exists; nothing to do")
        return 0
    run("npm", "ci", "--no-audit", "--no-fund", "--ignore-scripts")
    run("npm", "run", "build")
    tarball = ROOT / out("npm", "pack", "--silent").splitlines()[-1]
    print(f"packed {tarball.name} ({tarball.stat().st_size} bytes)")
    if dry:
        print("dry run: not publishing")
        return 0
    sha = out("git", "rev-parse", "HEAD")
    run(
        "gh", "release", "create", tag, str(tarball), "--target", sha, "--prerelease",
        "--title", f"playcanvas {version}",
        "--notes", f"Fork build of {sha}. Consumed by norna-game as a tarball dependency.",
    )
    return 0


raise SystemExit(main())
