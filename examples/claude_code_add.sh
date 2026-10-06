#!/usr/bin/env bash
# Register the server with Claude Code. Run from the repo folder, with the virtualenv created.
# Replace the two tokens first.
claude mcp add-json precedent "{
  \"type\": \"stdio\",
  \"command\": \"$(pwd)/.venv/bin/precedent-mcp\",
  \"args\": [\"--ledger\", \"$(pwd)/precedent_ledger.jsonl\"],
  \"env\": {\"TABPFN_TOKEN\": \"YOUR_PRIOR_LABS_KEY\", \"HF_TOKEN\": \"YOUR_HF_TOKEN\"}
}"
