#!/bin/sh
set -eu
repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"
python3 - <<'PY'
from pathlib import Path
import hashlib
p=Path('vendor/glpk-4.44.tar.gz')
expected='289a4366041ee7360b5523dbfdcb083487743357b96529e4af2477a65a3c7e6d'
assert hashlib.sha256(p.read_bytes()).hexdigest()==expected, 'GLPK source checksum mismatch'
PY
mkdir -p build/native
if [ ! -d build/native/glpk-4.44 ]; then
    tar -xzf vendor/glpk-4.44.tar.gz -C build/native
fi
cd build/native/glpk-4.44
./configure --disable-shared --enable-static
make -j "${BUILD_JOBS:-2}"
cd "$repo_root"
${CC:-cc} -O2 -Ibuild/native/glpk-4.44/include vendor/diagnostic/glpk444_text.c build/native/glpk-4.44/src/.libs/libglpk.a -lm -o vendor/diagnostic/glpk444_text
python3 - <<'PY'
from pathlib import Path
import hashlib,json,platform
p=Path('vendor/diagnostic/glpk444_text')
record={'platform':platform.platform(),'solver':'GLPK 4.44 (standalone)','binary_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
Path('build/native/solver-build.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
PY
