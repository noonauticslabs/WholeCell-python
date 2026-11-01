"""Create a deterministic source bundle from the reviewed repository file set."""

from pathlib import Path
import gzip, hashlib, io, json, os, subprocess, sys, tarfile

ROOT = Path(__file__).resolve().parents[1]
NAME = "wholecell-py-0.1.0a1"


def main():
    inventory = ROOT / "validation/release-files.json"
    if (ROOT / ".git").exists():
        names = [
            p.decode()
            for p in subprocess.check_output(
                ["git", "ls-files", "-z"], cwd=ROOT
            ).split(b"\0")
            if p
        ]
    else:
        names = [x["path"] for x in json.loads(inventory.read_text())["files"]]
    names = sorted(set(names) - {"validation/release-files.json"})
    records = [
        dict(
            path=n,
            sha256=hashlib.sha256((ROOT / n).read_bytes()).hexdigest(),
            bytes=(ROOT / n).stat().st_size,
        )
        for n in names
    ]
    inventory.write_text(
        json.dumps(dict(format_version=1, files=records), indent=2) + "\n"
    )
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/check_release.py")],
        cwd=ROOT,
        check=True,
    )
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    target = dist / (NAME + "-source.tar.gz")
    with (
        target.open("wb") as f,
        gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0) as gz,
        tarfile.open(fileobj=gz, mode="w") as tar,
    ):
        for name in sorted(names + ["validation/release-files.json"]):
            p = ROOT / name
            data = p.read_bytes()
            info = tarfile.TarInfo(NAME + "/" + name)
            info.size = len(data)
            info.mode = 0o755 if os.access(p, os.X_OK) else 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            tar.addfile(info, io.BytesIO(data))
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    target.with_suffix(target.suffix + ".sha256").write_text(
        digest + "  " + target.name + "\n"
    )
    print(
        json.dumps(
            dict(
                archive=str(target.relative_to(ROOT)),
                bytes=target.stat().st_size,
                sha256=digest,
                files=len(names) + 1,
            )
        )
    )


if __name__ == "__main__":
    main()
