"""Expand checked model fixtures without downloading data or overwriting edits."""

from pathlib import Path
import gzip
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / "reference/manifest.json").read_text())
    for row in manifest["files"]:
        target = ROOT / "reference" / row["path"]
        if row.get("archive"):
            archive = ROOT / "reference" / row["archive"]
            assert (
                hashlib.sha256(archive.read_bytes()).hexdigest()
                == row["archive_sha256"]
            ), archive.name
            if not target.exists():
                data = gzip.decompress(archive.read_bytes())
                assert (
                    hashlib.sha256(data).hexdigest() == row["sha256"]
                ), target.name
                with target.open("xb") as f:
                    f.write(data)
        assert target.is_file(), f"Missing fixture: {target.name}"
        assert (
            hashlib.sha256(target.read_bytes()).hexdigest() == row["sha256"]
        ), f"Modified fixture: {target.name}"
    print(f"Verified {len(manifest['files'])} fixtures")


if __name__ == "__main__":
    main()
