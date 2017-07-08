#! /bin/sh
# Convert binary to binary-text.
# Command line:
#     convert-binary-to-text.sh xxx.cl.bin xxx.clx.bin
#
# Usage of binary-text file (if needed):
# 1. generate binary file, related script: libxcam/tests/test-binary-kernel
#    $ test-binary-kernel --src-kernel kernel_demo.cl --bin-kernel kernel_demo.cl.bin --kernel-name kernel_demo
#
# 2. generate binary-text file, related script: libxcam/tools/convert-binary-to-text.sh
#    $ convert-binary-to-text.sh kernel_demo.cl.bin kernel_demo.clx.bin
#
# 3. include binary-text file when create image handler, please refer to demo handler:
#    SmartPtr<CLImageHandler> create_cl_binary_demo_image_handler (SmartPtr<CLContext> &context)


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
