#!/bin/sh
# Probe: /api/backup/create 500s for tenant "portll" while reads work. 2026-08-05.
# Real endpoint, real key (from Veld Desktop registry). Exits 0 on completion.
KEY=$(node -e 'console.log(require("/Users/portll/Repositories/Portll/veld-desktop/lib/registry.cjs").readApiKey())')
URL=http://127.0.0.1:3030

for user in portll sleight; do
  echo "== backup/create user_id=$user =="
  curl -s -w '\nHTTP %{http_code}\n' -m 30 -X POST "$URL/api/backup/create" \
    -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
    -d "{\"user_id\":\"$user\"}"
done

echo "== control read: backup/list portll =="
curl -s -w '\nHTTP %{http_code}\n' -m 10 -X POST "$URL/api/backup/list" \
  -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
  -d '{"user_id":"portll"}' | head -c 600
exit 0
