#!/usr/bin/env bash
# Online backup of the persistent candle store (t_18e7517a).
# History is UNRECOVERABLE past HL retention, so the db itself is the asset:
# zstd snapshots under archive/candlestore/, newest N kept. sqlite3 backup API
# (online-safe under WAL; no VACUUM needed — blob pages copy as-is).
set -euo pipefail
cd "$(dirname "$0")/.."

SRC="${HERMES_CANDLESTORE_DIR:-candlestore}/candles.db"
DEST_DIR="archive/candlestore"
KEEP=7
[ -f "$SRC" ] || { echo "[candlestore-backup] no store at $SRC — nothing to do"; exit 0; }
mkdir -p "$DEST_DIR"
STAMP=$(date -u +%Y%m%d-%H%M)
TMP="$DEST_DIR/.candles-$STAMP.db"
DEST="$DEST_DIR/candles-$STAMP.db.zst"

python3 - "$SRC" "$TMP" <<'EOF'
import sqlite3, sys
src, dst = sys.argv[1], sys.argv[2]
s = sqlite3.connect(src); d = sqlite3.connect(dst)
with s:
    s.backup(d)          # online backup API — safe while the bot writes
d.close(); s.close()
EOF

zstd -q -f -o "$DEST" "$TMP"
rm -f "$TMP"
# verify the snapshot opens + row count
ROWS=$(python3 - "$DEST" <<'EOF'
import sqlite3, subprocess, sys, tempfile, os
data = subprocess.run(["zstd", "-dc", sys.argv[1]], capture_output=True, check=True).stdout
tmpf = tempfile.NamedTemporaryFile(suffix=".db", delete=False); tmpf.write(data); tmpf.close()
con = sqlite3.connect(tmpf.name)
print(con.execute("PRAGMA integrity_check").fetchone()[0],
      con.execute("SELECT COUNT(*) FROM bars").fetchone()[0], file=sys.stderr)
print(con.execute("SELECT COUNT(*) FROM bars").fetchone()[0])
con.close(); os.unlink(tmpf.name)
EOF
)
echo "[candlestore-backup] $DEST rows=$ROWS"
ls -1t "$DEST_DIR"/candles-*.db.zst | tail -n +$((KEEP+1)) | xargs -r rm -f
echo "[candlestore-backup] kept newest $KEEP in $DEST_DIR"
