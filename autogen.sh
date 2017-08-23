#!/bin/sh

pre_commit_hook=".git/hooks/pre-commit"
if test ! -L $pre_commit_hook;
then
    rm -rf $pre_commit_hook
    ln -s ../../tools/pre-commit-code-style.sh $pre_commit_hook
    echo "link $pre_commit_hook to code style check"
fi

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
