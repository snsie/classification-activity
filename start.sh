#!/usr/bin/env bash
# Start the activity app (or the website) for a local look.
#
#   ./start.sh                      starter Streamlit app
#   ./start.sh solution             private full-solution Streamlit app
#   ./start.sh site                 the student website (builds it first)
#   ./start.sh ... --port 9000      choose the port (default: the first free one from 8765)
#
# Press Ctrl+C to stop. On a remote machine, forward the port (VS Code does this from the Ports tab).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$ROOT/.venv/bin/python"
TARGET="starter"
PORT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    starter|solution|site) TARGET="$1" ;;
    --port) PORT="${2:?--port needs a number}"; shift ;;
    -h|--help) sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown option: $1 (try ./start.sh --help)" >&2; exit 1 ;;
  esac
  shift
done

[[ -x "$PY" ]] || { echo "No .venv found. Create it first (see README.md, Tests)." >&2; exit 1; }

if [[ -z "$PORT" ]]; then
  PORT="$("$PY" - <<'EOF'
import socket
for port in range(8765, 8865):
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            continue
        print(port)
        break
EOF
)"
fi

case "$TARGET" in
  starter) DIR="$ROOT/starter" ;;
  solution)
    DIR="$ROOT/instructor/solution/judge-the-model"
    [[ -d "$DIR" ]] || { echo "No solution yet. Build it: $PY instructor/make_solution.py" >&2; exit 1; }
    ;;
  site)
    cd "$ROOT"
    node scripts/build.mjs
    echo "Website: http://localhost:$PORT/classification-activity/"
    exec node scripts/serve.mjs "$PORT"
    ;;
esac

cd "$DIR"
echo "Streamlit app ($TARGET): http://localhost:$PORT/"
exec "$PY" -m streamlit run streamlit_app.py --server.port "$PORT" --server.headless true
