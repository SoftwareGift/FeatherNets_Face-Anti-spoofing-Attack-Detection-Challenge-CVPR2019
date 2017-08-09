# XCAM_MD5SUM([file], [md5sum], [if-true], [if-false])
AC_DEFUN([XCAM_MD5SUM],
[
    AS_IF([test -f $1],
        [
            md5=`md5sum $1 | cut --delimiter=' ' --fields=1`
            AS_IF([test "x$md5" = "x$2"], [$3], [$4])
        ],
        [$4])
])

# XCAM_WGET([url], [output-file], [md5sum])
AC_DEFUN([XCAM_WGET],
[
    MD5_CORRECT=yes
    XCAM_MD5SUM([$2], [$3], [MD5_CORRECT=yes], [MD5_CORRECT=no])
    AS_IF([test "x$MD5_CORRECT" = "xyes"],
        [AC_MSG_NOTICE([checking $2 md5sum ... ok])],
        [
            AC_MSG_NOTICE([downloading $2...])
            dir=`dirname $2`
            AS_IF([test ! -d $dir], [mkdir -p $dir])

            wget --tries=2 --timeout=5 -q --no-use-server-timestamps $1 -O $2
            AS_IF([test "$?" != 0], [AC_MSG_ERROR([download/wget $2 failed])])

            XCAM_MD5SUM([$2], [$3], [AC_MSG_NOTICE([checking $2 md5sum ... ok])], [AC_MSG_ERROR([checking $2 md5sum ... failed])])
        ])
])

