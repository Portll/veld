#!/bin/sh
# veld's embedder on its own llama-server, so embedding does not queue behind whatever else an LLM
# host is serving. Configured by ~/.config/veld/embedder.env (or $VELD_EMBEDDER_ENV):
#
#   EMBEDDER_MODEL_PATH   the GGUF file (required)
#   EMBEDDER_ALIAS        the id /v1/models lists; must equal VELD_EMBEDDING_API_MODEL (required)
#   EMBEDDER_WINDOW       tokens per input, the model's trained window (default 2048)
#   EMBEDDER_SLOTS        inputs served at once (default 4)
#   EMBEDDER_POOLING      mean for nomic, cls for bge-m3 (default mean)
#   EMBEDDER_PORT         default 8095
#   EMBEDDER_API_KEY      when set, every request needs it as a Bearer token
#   EMBEDDER_EXTRA_ARGS   passed through to llama-server
#   LLAMA_SERVER          default /opt/homebrew/bin/llama-server
#
# llama-server divides -c among its slots, and an embedding input must fit in one micro-batch, so
# -c is the window times the slots and -b and -ub are the window. With its defaults (-ub 512 and
# -c split four ways) any input over 512 tokens is refused.
set -eu

ENV_FILE="${VELD_EMBEDDER_ENV:-$HOME/.config/veld/embedder.env}"
if [ -f "$ENV_FILE" ]; then
    . "$ENV_FILE"
fi

: "${EMBEDDER_MODEL_PATH:?set EMBEDDER_MODEL_PATH in $ENV_FILE}"
: "${EMBEDDER_ALIAS:?set EMBEDDER_ALIAS in $ENV_FILE}"
WINDOW="${EMBEDDER_WINDOW:-2048}"
SLOTS="${EMBEDDER_SLOTS:-4}"

set -- -m "$EMBEDDER_MODEL_PATH" --alias "$EMBEDDER_ALIAS" \
    --embedding --pooling "${EMBEDDER_POOLING:-mean}" \
    -c "$((WINDOW * SLOTS))" -np "$SLOTS" -b "$WINDOW" -ub "$WINDOW" \
    --host 127.0.0.1 --port "${EMBEDDER_PORT:-8095}"
if [ -n "${EMBEDDER_API_KEY:-}" ]; then
    set -- "$@" --api-key "$EMBEDDER_API_KEY"
fi
# shellcheck disable=SC2086 # EMBEDDER_EXTRA_ARGS is a list of words by design
exec "${LLAMA_SERVER:-/opt/homebrew/bin/llama-server}" "$@" ${EMBEDDER_EXTRA_ARGS:-}
