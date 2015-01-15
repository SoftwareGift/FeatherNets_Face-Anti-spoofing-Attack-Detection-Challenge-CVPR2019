#!/bin/sh
echo "Generating configure files"
autoreconf -i
# Run twice to get around a "ltmain.sh" bug
autoreconf --install --force
