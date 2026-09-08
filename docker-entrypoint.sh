#!/bin/sh
# Named volumes mount as root; ensure HF cache is writable by app (uid 1000).
set -e
mkdir -p /home/app/.cache/huggingface
chown -R app:app /home/app/.cache
exec runuser -u app -- "$@"
