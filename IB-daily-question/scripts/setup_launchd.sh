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

    # Unload if loaded, then load (modern bootstrap/bootout commands)
    launchctl bootout "gui/$UID" "$DST" 2>/dev/null || true
    launchctl bootstrap "gui/$UID" "$DST"
    echo "Loaded $name"
done

echo
echo "Both jobs installed. Verify with:"
echo "  launchctl list | grep ib-daily"
echo
echo "To uninstall later:"
echo "  launchctl bootout gui/\$UID $TARGET_DIR/com.chrislane.ib-daily.send.plist"
echo "  launchctl bootout gui/\$UID $TARGET_DIR/com.chrislane.ib-daily.listener.plist"
echo "  rm $TARGET_DIR/com.chrislane.ib-daily.*.plist"
