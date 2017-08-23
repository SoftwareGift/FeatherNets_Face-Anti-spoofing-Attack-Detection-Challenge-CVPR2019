#!/bin/sh
# checking code style before commit

ASTYLE=astyle
ASTYLE_PARAMS="--indent=spaces=4 --convert-tabs --pad-oper --suffix=none"

DOS2UNIX=dos2unix
DOS2UNIX_PARAMS="-ascii --safe --keepdate --quiet"

command -v $ASTYLE > /dev/null 2>&1 || echo "warning: $ASTYLE is not installed"
command -v $DOS2UNIX > /dev/null 2>&1 || echo "warning: $DOS2UNIX is not installed"

echo "---- checking code style (dos2unix / astyle)----"
for file in `git diff-index --cached --name-only HEAD --diff-filter=ACMR | grep -E "\.c$|\.cpp$|\.h$|\.cl$|\.hpp$" ` ; do
    $DOS2UNIX ${DOS2UNIX_PARAMS} ${file}
    $ASTYLE ${ASTYLE_PARAMS} ${file}
    ret=$?
    if [ $ret != 0 ] ; then
        echo "code style failed on $file"
        exit 1
    fi
    git add $file
done
echo "---- checking code style done----"
