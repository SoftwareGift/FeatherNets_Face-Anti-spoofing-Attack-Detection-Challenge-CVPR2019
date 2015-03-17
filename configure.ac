#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

m4_define([xcam_major_version], [0])
m4_define([xcam_minor_version], [1])
m4_define([xcam_micro_version], [0])
m4_define([xcam_version], [xcam_major_version.xcam_minor_version.xcam_micro_version])

AC_PREREQ([2.60])
AC_CONFIG_MACRO_DIR([m4])
AC_INIT([libxcam], [xcam_version], [feng.yuan@intel.com], [libxcam])
AC_CONFIG_HEADERS([config.h])
AM_INIT_AUTOMAKE

#xcam version
XCAM_MAJOR_VERSION=xcam_major_version
XCAM_MINOR_VERSION=xcam_minor_version
XCAM_MICRO_VERSION=xcam_micro_version
XCAM_VERSION=xcam_version
XCAM_VERSION_HEX="0x$XCAM_MAJOR_VERSION$XCAM_MINOR_VERSION$XCAM_MICRO_VERSION"
AC_DEFINE_UNQUOTED(XCAM_VERSION, $XCAM_VERSION_HEX,
    [define libxcam version])
AC_SUBST(XCAM_VERSION)

# Checks for programs.
AC_PROG_CC
AC_PROG_CXX
AC_PROG_INSTALL
LT_INIT

AC_ARG_ENABLE(debug,
    AS_HELP_STRING([--enable-debug],
                   [enable debug, @<:@default=no@:>@]),
    [], [enable_debug="no"])
AM_CONDITIONAL([DEBUG], [test "$enable_debug" = "yes"])

AC_ARG_ENABLE(drm,
    AS_HELP_STRING([--enable-drm],
                   [enable drm buffer, @<:@default=yes@:>@]),
    [], [enable_drm="yes"])

AC_ARG_ENABLE([aiq],
    AS_HELP_STRING([--enable-aiq],
                   [enable Aiq 3A algorithm build, @<:@default=no@:>@]),
    [], [enable_aiq="no"])

AC_ARG_ENABLE(libcl,
    AS_HELP_STRING([--enable-libcl],
                   [enable libcl image processor, @<:@default=yes@:>@]),
    [], [enable_libcl="yes"])

# documentation
AC_ARG_ENABLE(docs,
    [AC_HELP_STRING([--enable-docs],
                    [build Doxygen documentation @<:@default=no@:>@])],
    [], [enable_docs="no"])

# Check for Doxygen
if test "$enable_docs" = "yes"; then
    AC_CHECK_TOOL([DOXYGEN], [doxygen], [no])
    if test "$DOXYGEN" = "no"; then
       enable_docs="no"
    fi
fi
AM_CONDITIONAL(ENABLE_DOCS, test "$enable_docs" = "yes")

# check drm
HAVE_LIBDRM=0
if test "$enable_drm" = "yes"; then
    PKG_CHECK_MODULES(LIBDRM, [libdrm], [HAVE_LIBDRM=1], [HAVE_LIBDRM=0])
fi

# check libcl
HAVE_LIBCL=0
if test "$enable_libcl" = "yes"; then
    PKG_CHECK_MODULES(LIBCL, [libcl], [HAVE_LIBCL=1], [HAVE_LIBCL=0])
fi

# check AIQ
HAVE_IA_AIQ=0
USE_LOCAL_AIQ=0
if test "$enable_aiq" = "yes"; then
    HAVE_IA_AIQ=1
    PKG_CHECK_MODULES(IA_AIQ, [libia_aiq], [USE_LOCAL_AIQ=0], [USE_LOCAL_AIQ=1])
	# installed in default PATH
    # AC_CHECK_HEADERS([ia_aiq.h], [HAVE_IA_AIQ=1], [:])
fi

# check atomisp headers
USE_LOCAL_ATOMISP=0
#AC_CHECK_HEADERS([linux/atomisp.h], [USE_LOCAL_ATOMISP=0], [USE_LOCAL_ATOMISP=1])
AC_CACHE_CHECK([for linux/atomisp.h],
    ac_cv_have_atomisp_headers, [
    saved_CPPFLAGS="$CPPFLAGS"
    CPPFLAGS="$CPPFLAGS"
    saved_LIBS="$LIBS"
    LIBS="$LIBS"
    AC_COMPILE_IFELSE(
        [AC_LANG_PROGRAM(
            [[#ifndef __user
              #define __user 
              #endif
              #include <stdint.h>
              #include <stdio.h>
              #include <linux/atomisp.h>]],
            [[struct atomisp_parm param;]]
           )],
        [ac_cv_have_atomisp_headers="yes"],
        [ac_cv_have_atomisp_headers="no" USE_LOCAL_ATOMISP=1]
    )
    CPPFLAGS="$saved_CPPFLAGS"
    LIBS="$saved_LIBS"
])


dnl set XCAM_CFLAGS and XCAM_CXXFLAGS
XCAM_CFLAGS=" -fPIC -DSTDC99 -W -Wall -D_REENTRANT"
if test "$enable_debug" = "yes"; then
    XCAM_CFLAGS="$XCAM_CFLAGS  -g -DDEBUG"
fi
XCAM_CXXFLAGS="$XCAM_CFLAGS -std=c++0x"
AC_SUBST(XCAM_CFLAGS)
AC_SUBST(XCAM_CXXFLAGS)

# define macor in config.h
AC_DEFINE_UNQUOTED([HAVE_LIBDRM], $HAVE_LIBDRM,
    [have libdrm])
AM_CONDITIONAL([HAVE_LIBDRM], [test "$HAVE_LIBDRM" -eq 1])

AC_DEFINE_UNQUOTED([HAVE_LIBCL], $HAVE_LIBCL,
    [have libcl])
AM_CONDITIONAL([HAVE_LIBCL], [test "$HAVE_LIBCL" -eq 1])

#atomisp
AM_CONDITIONAL([USE_LOCAL_ATOMISP], [test "$USE_LOCAL_ATOMISP" -eq 1])

# aiq (ia_imaging)
AC_DEFINE_UNQUOTED([HAVE_IA_AIQ], $HAVE_IA_AIQ,
    [have aiq binary])
AM_CONDITIONAL([HAVE_IA_AIQ], [test "$HAVE_IA_AIQ" -eq 1])
AM_CONDITIONAL([USE_LOCAL_AIQ], [test "$USE_LOCAL_AIQ" -eq 1])

AC_CONFIG_FILES([Makefile
                 xcore/Makefile
                 wrapper/Makefile
                 wrapper/gstreamer/Makefile
                 tests/Makefile
                 pkgconfig/Makefile
                 pkgconfig/xcam_core.pc
                ])
 
AC_OUTPUT

if test "$USE_LOCAL_AIQ" -eq 1; then use_local_aiq="yes"; else  use_local_aiq="no"; fi
if test "$USE_LOCAL_ATOMISP" -eq 1; then use_local_atomisp="yes"; else  use_local_atomisp="no"; fi
if test "$HAVE_LIBCL" -eq 1; then have_libcl="yes"; else  have_libcl="no"; fi

echo "
     libxcam configuration summary
     version                    : $XCAM_VERSION
     enable debug               : $enable_debug
     build aiq analyzer         : $enable_aiq
     use local aiq              : $use_local_aiq
     use local atomisp          : $use_local_atomisp
     have opencl lib            : $have_libcl
"
