#! /bin/sh
# Convert binary to binary-text.
# Command line:
#     convert-binary-to-text.sh xxx.cl.bin xxx.clx.bin

BINARY_FILE=$1
TEXT_FILE=$2

if [ $# -ne 2 ]; then
    echo "Usage: $0 <binary_file> <text_file>"
    exit 1
fi

od -A n -t x1 -v $BINARY_FILE | \
    gawk '
    BEGIN { print "{" }
    {
        printf "   "
        for (i = 1; i < NF; i++)
            { printf " 0x" $i "," }
        print " 0x" $i ","
    }
    END { print "};" }
    ' > $TEXT_FILE.tmp

ret=$?
if [ $ret != 0 ]; then
    echo "Convert $BINARY_FILE to $TEXT_FILE faild"
    rm -f $TEXT_FILE.tmp
    exit 1
fi

mv $TEXT_FILE.tmp $TEXT_FILE

echo "Convert $BINARY_FILE to $TEXT_FILE done"
