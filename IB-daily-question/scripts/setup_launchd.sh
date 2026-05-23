#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
TARGET_DIR="$HOME/Library/LaunchAgents"
mkdir -p "$TARGET_DIR" "$REPO/data/logs"

for name in com.chrislane.ib-daily.send com.chrislane.ib-daily.listener; do
    SRC="$REPO/launchd/$name.plist"
    DST="$TARGET_DIR/$name.plist"
    sed "s#__REPO__#$REPO#g" "$SRC" > "$DST"
    echo "Installed $DST"

    # Unload if loaded, then load
    launchctl unload "$DST" 2>/dev/null || true
    launchctl load "$DST"
    echo "Loaded $name"
done

echo
echo "Both jobs installed. Verify with:"
echo "  launchctl list | grep ib-daily"
echo
echo "To uninstall later:"
echo "  launchctl unload $TARGET_DIR/com.chrislane.ib-daily.*.plist"
echo "  rm $TARGET_DIR/com.chrislane.ib-daily.*.plist"
