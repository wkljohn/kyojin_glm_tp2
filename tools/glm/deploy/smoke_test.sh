#!/usr/bin/env bash
# Smoke test of a running EP server: models, greedy chat, sampled chat, tool call, streamed chat.
#   ./smoke_test.sh [base url=http://127.0.0.1:8083]   (add API_KEY=... when going through an auth proxy)
set -uo pipefail
U="${1:-http://127.0.0.1:8083}"
H=(-H 'content-type: application/json'); [ -n "${API_KEY:-}" ] && H+=(-H "Authorization: Bearer $API_KEY")
M=$(curl -sf "${H[@]}" "$U/v1/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])') || { echo "FAIL /v1/models"; exit 1; }
echo "model: $M"
fail=0
q() {
    curl -sf "${H[@]}" "$U/v1/chat/completions" -d "$1" | python3 -c '
import json, sys
d = json.load(sys.stdin); c = d["choices"][0]; t = d.get("timings", {}); m = c["message"]
out = m.get("tool_calls") or repr(m.get("content"))[:120]
print("  %-10s pf %6.0f t/s  dec %5.1f t/s | %s" % (c["finish_reason"], t.get("prompt_per_second", 0), t.get("predicted_per_second", 0), out))
' || { echo "  FAIL"; fail=1; }
}
echo "greedy:";  q '{"model":"'$M'","messages":[{"role":"user","content":"What is 17*23? Answer briefly."}],"temperature":0,"max_tokens":400}'
echo "sampled:"; q '{"model":"'$M'","messages":[{"role":"user","content":"Write a haiku about RDMA."}],"temperature":0.8,"max_tokens":400}'
echo "tool:";    q '{"model":"'$M'","messages":[{"role":"user","content":"What is the weather in Paris?"}],"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],"temperature":0,"max_tokens":400}'
echo "stream:";  curl -sfN "${H[@]}" "$U/v1/chat/completions" -d '{"model":"'$M'","stream":true,"messages":[{"role":"user","content":"Count from 1 to 5."}],"temperature":0,"max_tokens":300}' | grep -c '^data:' | sed 's/^/  chunks: /' || fail=1
[ $fail = 0 ] && echo PASS || { echo FAIL; exit 1; }
