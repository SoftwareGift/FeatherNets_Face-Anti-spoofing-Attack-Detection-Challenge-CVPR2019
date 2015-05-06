#! /bin/sh

OLD_CL_FILE=$1
NEW_CL_FILE=$2

awk '
    BEGIN {FS=""}
    {
        if ($0~/^[\t " "]*[\/]+/ || $0~/^[\t " "]*[\*]/)
            printf("%s\n", $0);
        else
        {
            gsub(/^\"[\\]?n?/, "");
            gsub(/[\\]?n?\"[ ]*$/, "");

            if ($0~/^[ ]*$/)
                printf("\n");
            else
            {
                gsub(/\\\"/, "\"");
                gsub(/\\%/, "%");
                gsub(/\\\\n/, "\\n");

                gsub(/\"/, "\\\"");
                gsub(/%/, "\\%");
                gsub(/\\n/, "\\\\n");

                gsub(/^#/, "\\n#");

                printf("\"%s\\n\"\n", $0);
            }
        }
    }
    ' $OLD_CL_FILE > $OLD_CL_FILE.tmp

ret=$?
if [ $ret != 0 ] ; then
    rm -rf $OLD_CL_FILE.tmp
    echo "add double quotation on $OLD_CL_FILE failed"
else
    mv $OLD_CL_FILE.tmp $NEW_CL_FILE
    echo "add double quotation on $OLD_CL_FILE done"
fi

