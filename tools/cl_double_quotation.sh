#! /bin/sh
# add double quotation marks on cl file, this script will 
# be called in /libxcam/clx_kernel/Makefile

OLD_CL_FILE=$1
NEW_CL_FILE=$2

awk '
    BEGIN {FS=""}
    {
        if ($0~/^[\t " "]*[\/]+/ || $0~/^[\t " "]*[\*]/)
            printf("%s\n", $0);
        else
        {
            if ($0~/^[ ]*$/)
                printf("\n");
            else
            {
                gsub(/\"/, "\\\"");
                gsub(/%/, "\\%");
                gsub(/\\n/, "\\\\n");
                gsub(/\\t/, "\\\\t");

                gsub(/^#/, "\\n#");

                printf("\"%s\\n\"\n", $0);
            }
        }
    }
    ' $OLD_CL_FILE > $NEW_CL_FILE.tmp

ret=$?
if [ $ret != 0 ] ; then
    rm -rf $NEW_CL_FILE.tmp
    echo "add double quotation marks on $OLD_CL_FILE failed"
else
    mv $NEW_CL_FILE.tmp $NEW_CL_FILE
    echo "add double quotation marks on $OLD_CL_FILE done"
fi

