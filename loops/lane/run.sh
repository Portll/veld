#!/bin/zsh
# Keeps one runner lane alive; it ends on STOP (every lane) or STOP-<host> (this lane). One
# supervisor per host: a second one exits at once.
cd "${0:A:h}"
HOST=${1:-local}
LOCK="/tmp/pli-qwen-$HOST.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  [ -f "$LOCK/pid" ] && kill -0 "$(cat $LOCK/pid)" 2>/dev/null && exit 0
  rm -rf "$LOCK"; mkdir "$LOCK" || exit 0
fi
echo $$ > "$LOCK/pid"
trap 'rm -rf "$LOCK"' EXIT
while [ ! -f STOP ] && [ ! -f "STOP-$HOST" ]; do
  # After a restart LM Studio may come up without the model.
  if [ "$HOST" = local ] && ! ~/.lmstudio/bin/lms ps 2>/dev/null | grep -q "qwen/qwen3.8-27b"; then
    ~/.lmstudio/bin/lms load qwen/qwen3.8-27b -c 100096 --parallel 1 --gpu max -y > /dev/null 2>&1
  fi
  node runner.mjs --host "$HOST" >> "runner-$HOST.out" 2>&1
  sleep 30
done
