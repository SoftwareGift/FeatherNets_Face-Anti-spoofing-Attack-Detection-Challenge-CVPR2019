#!/bin/sh

ln -s ../../tools/pre-commit-code-style.sh .git/hooks/pre-commit

echo "git submodule update"
git submodule sync
git submodule init
git submodule update

echo "Generating configure files"
autoreconf -i
# Run twice to get around a "ltmain.sh" bug
autoreconf --install --force

srcdir=`dirname "$0"`
test -z "$srcdir" && srcdir=.

if test -z "$NOCONFIGURE"; then
    $srcdir/configure "$@"
fi
