#!/bin/sh
echo "git submodule update"
git submodule sync
git submodule init
git submodule update

echo "Generating configure files"
autoreconf -i
# Run twice to get around a "ltmain.sh" bug
autoreconf --install --force
