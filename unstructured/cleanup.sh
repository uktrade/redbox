#!/bin/sh
set -e

TARGET_DIR="/tmp/unstructured_temp"
AGE_MINUTES=15

if [ ! -d "$TARGET_DIR" ]; then
    exit 0
fi

DELETED_COUNT=$(find "$TARGET_DIR" \
    -xdev \
    -mindepth 1 \
    -type f \
    -mmin +$AGE_MINUTES \
    -print -delete | wc -l)

if [ "$DELETED_COUNT" -gt 0 ]; then
    echo "Deleted $DELETED_COUNT files older than ${AGE_MINUTES} minutes from $TARGET_DIR"
fi
