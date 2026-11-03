"""Check the publishable file set, local links, runtime identity and sensitive debris."""

from pathlib import Path
import gzip, hashlib, json, re, subprocess, sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wholecell_py.checkpoint import source_digest


def release_bytes(path):
    """Return platform-independent bytes for a checked-in release file.

    Git normalizes text files to LF in the repository, but a checkout with
    core.autocrlf can expose CRLF in the working tree. Hash the canonical text
    representation so the same release inventory validates on every OS.
    Binary files are returned byte-for-byte.
    """
    data = path.read_bytes()
    if b"\0" not in data:
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            pass
        else:
            return data.replace(b"\r\n", b"\n")
    return data


def main():
    inventory = ROOT / "validation/release-files.json"
    if inventory.exists():
        entries = json.loads(inventory.read_text())["files"]
        paths = [ROOT / entry["path"] for entry in entries]
        for entry, p in zip(entries, paths):
            assert p.is_file(), entry["path"]
            assert (
                hashlib.sha256(release_bytes(p)).hexdigest() == entry["sha256"]
            ), ("Changed release file: " + entry["path"])
    else:
        result = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
        paths = [ROOT / p.decode() for p in result.stdout.split(b"\0") if p]
    assert paths, "No publishable files"
    forbidden_parts = {
        ".venv",
        "__pycache__",
        ".pytest_cache",
        "results",
        ".ssh",
        "node_modules",
    }
    patterns = {
        "private key": rb"-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----",
        "GitHub credential": rb"(?:ghp_|github_pat_)[A-Za-z0-9_]{30,}",
        "private host path": rb"/(?:Users|home)/[A-Za-z0-9_.-]+/",
        "API credential": rb"sk-[A-Za-z0-9_-]{32,}",
    }
    issues = []
    for p in paths:
        rel = p.relative_to(ROOT)
        if forbidden_parts.intersection(rel.parts) or p.suffix in {
            ".pkl",
            ".pyc",
            ".log",
        }:
            issues.append((str(rel), "private/generated file"))
        data = release_bytes(p)
        if p.name.endswith(".json.gz"):
            data = gzip.decompress(data)
        if p.suffix in {
            ".py",
            ".md",
            ".json",
            ".m",
            ".txt",
            ".toml",
            ".yml",
            ".cff",
            ".csv",
        } or p.name.endswith(".json.gz"):
            for label, pattern in patterns.items():
                if re.search(pattern, data):
                    issues.append((str(rel), label))
        if p.suffix == ".md":
            for target in re.findall(
                r"!?\[[^\]]*\]\(([^)]+)\)", data.decode()
            ):
                if re.match(r"[a-z]+://|#|mailto:", target):
                    continue
                dest = (p.parent / target.split("#")[0]).resolve()
                if not dest.exists():
                    issues.append((str(rel), "broken local link: " + target))
    assert not issues, json.dumps(issues, indent=2)
    expected = json.loads(
        (ROOT / "validation/source-manifest.json").read_text()
    )["runtime_source_digest"]
    assert (
        source_digest() == expected
    ), "Runtime hash differs from validation candidate"
    print(
        json.dumps(
            dict(
                passed=True,
                files_checked=len(paths),
                runtime_source_digest=expected,
                scope="Static selected-file scan, relative links and runtime identity; not a guarantee that arbitrary secrets cannot be present",
            )
        )
    )


if __name__ == "__main__":
    main()
