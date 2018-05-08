#! /bin/sh
# Add double quotation marks on source file
# Usage: add-quotation-marks.sh <src_file> <dst_file>

SRC_FILE=$1
DST_FILE=$2

if [ $# -ne 2 ]; then
    echo "Usage: $0 <src_file> <dst_file>"
    exit 1
fi

gawk '
    BEGIN { FS = "" }
    {
        if ($0~/^[\t " "]*[\/]+/ || $0~/^[\t " "]*[\*]/)
            print $0
        else
        {
            if ($0~/^[ ]*$/)
                print
            else
            {
                $0 = gensub (/\\$/, "\\\\\\\\", "g")
                $0 = gensub (/\"/, "\\\\\\\"", "g")
                $0 = gensub (/%/, "\\\\%", "g")
                $0 = gensub (/\\n/, "\\\\\\\\n", "g")
                $0 = gensub (/\\t/, "\\\\\\\\t", "g")
                $0 = gensub (/^#/, "\\\\n#", "g")

                print "\""$0"\\n\""
            }
        }
    }
    ' $SRC_FILE > $DST_FILE.tmp

ret=$?
if [ $ret != 0 ]; then
    rm -rf $DST_FILE.tmp
    echo "Add double quotation marks on $SRC_FILE failed"
    exit 1
fi

mv $DST_FILE.tmp $DST_FILE

echo "Add double quotation marks on $SRC_FILE done"
