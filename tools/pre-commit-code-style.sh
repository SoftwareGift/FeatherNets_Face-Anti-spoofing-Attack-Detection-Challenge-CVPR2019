#!/bin/sh
# checking code style before commit

ASTYLE=astyle
ASTYLE_PARMS="--indent=spaces=4 --convert-tabs --pad-oper --suffix=none"

echo "---- checking code style ----"
for file in `git diff-index --cached --name-only HEAD --diff-filter=ACMR | grep -E "\.c$|\.cpp$|\.h$|\.cl$" ` ; do
    $ASTYLE ${ASTYLE_PARMS} ${file}
    ret=$?
    if [ $ret != 0 ] ; then
        echo "code style failed on $file"
        exit 1
    fi
    git add $file
done
echo "---- checking code style done----"
