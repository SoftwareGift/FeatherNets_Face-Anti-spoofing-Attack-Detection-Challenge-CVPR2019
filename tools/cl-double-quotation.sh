#! /bin/sh
# Add double quotation marks on cl file, this script will
# be called in top_srcdir/clx_kernel/Makefile.am

CL_FILE=$1
CLX_FILE=$2

if [ $# -ne 2 ]; then
    echo "Usage: $0 <cl_file> <clx_file>"
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
    ' $CL_FILE > $CLX_FILE.tmp

ret=$?
if [ $ret != 0 ]; then
    rm -rf $CLX_FILE.tmp
    echo "Add double quotation marks on $CL_FILE failed"
    exit 1
fi

mv $CLX_FILE.tmp $CLX_FILE

echo "Add double quotation marks on $CL_FILE done"
