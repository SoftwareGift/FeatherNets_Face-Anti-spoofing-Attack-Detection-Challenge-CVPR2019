#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

m4_define([xcam_major_version], [1])
m4_define([xcam_minor_version], [0])
m4_define([xcam_micro_version], [0])
m4_define([xcam_version], [xcam_major_version.xcam_minor_version.xcam_micro_version])

AC_PREREQ([2.60])
AC_CONFIG_MACRO_DIR([m4])
AC_INIT([libxcam], [xcam_version], [feng.yuan@intel.com], [libxcam])
AC_CONFIG_HEADERS([config.h])
AM_INIT_AUTOMAKE([foreign])

#xcam version
XCAM_MAJOR_VERSION=xcam_major_version
XCAM_MINOR_VERSION=xcam_minor_version
XCAM_MICRO_VERSION=xcam_micro_version
XCAM_VERSION=xcam_version
XCAM_VERSION_HEX="0x$XCAM_MAJOR_VERSION$XCAM_MINOR_VERSION$XCAM_MICRO_VERSION"
AC_DEFINE_UNQUOTED(XCAM_VERSION, $XCAM_VERSION_HEX,
    [define libxcam version])
AC_SUBST(XCAM_VERSION)

XCAM_LT_VERSION="xcam_major_version:xcam_minor_version:xcam_micro_version"
XCAM_LT_LDFLAGS="-version-number $XCAM_LT_VERSION"
AC_SUBST(XCAM_LT_VERSION)
AC_SUBST(XCAM_LT_LDFLAGS)

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

AC_ARG_ENABLE(profiling,
    AS_HELP_STRING([--enable-profiling],
                   [enable profiling, @<:@default=no@:>@]),
    [], [enable_profiling="no"])

AC_ARG_ENABLE(drm,
    AS_HELP_STRING([--enable-drm],
                   [enable drm buffer, @<:@default=yes@:>@]),
    [], [enable_drm="yes"])

AC_ARG_ENABLE([aiq],
    AS_HELP_STRING([--enable-aiq],
                   [enable Aiq 3A algorithm build, @<:@default=no@:>@]),
    [], [enable_aiq="no"])

AC_ARG_ENABLE([gst],
    AS_HELP_STRING([--enable-gst],
                   [enable gstreamer plugin build, @<:@default=no@:>@]),
    [], [enable_gst="no"])

AC_ARG_ENABLE(libcl,
    AS_HELP_STRING([--enable-libcl],
                   [enable libcl image processor, @<:@default=yes@:>@]),
    [], [enable_libcl="yes"])

AC_ARG_ENABLE(opencv,
    AS_HELP_STRING([--enable-opencv],
                   [enable opencv library, @<:@default=no@:>@]),
    [], [enable_opencv="no"])

AC_ARG_ENABLE(capi,
    AS_HELP_STRING([--enable-capi],
                   [enable libxcam-capi library, @<:@default=yes@:>@]),
    [], [enable_capi="yes"])

# documentation
AC_ARG_ENABLE(docs,
    [AC_HELP_STRING([--enable-docs],
                    [build Doxygen documentation @<:@default=no@:>@])],
    [], [enable_docs="no"])

AC_ARG_ENABLE([3alib],
    AS_HELP_STRING([--enable-3alib],
                   [enable 3A lib build, @<:@default=no@:>@]),
    [], [enable_3alib="no"])

AC_ARG_ENABLE([smartlib],
    AS_HELP_STRING([--enable-smartlib],
                   [enable smart analysis lib build, @<:@default=no@:>@]),
    [], [enable_smartlib="no"])

# Check for Doxygen
if test "$enable_docs" = "yes"; then
    AC_CHECK_TOOL([DOXYGEN], [doxygen], [no])
    if test "$DOXYGEN" = "no"; then
       enable_docs="no"
    fi
fi
AM_CONDITIONAL(ENABLE_DOCS, test "$enable_docs" = "yes")

# check profiling
ENABLE_PROFILING=0
if test "$enable_profiling" = "yes"; then
    ENABLE_PROFILING=1
fi

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

if test "$enable_libcl" = "yes" && test "$HAVE_LIBCL" -eq 0; then
    PKG_CHECK_MODULES(LIBCL, [OpenCL], [HAVE_LIBCL=1], [HAVE_LIBCL=0])
fi

if test "$HAVE_LIBCL" -eq 1; then
    AC_CHECK_PROGS([GAWK], [gawk], [no])
    if test "x$GAWK" = "xno"; then
        AC_MSG_ERROR([gawk not found])
    fi
fi

# check opencv
HAVE_OPENCV=0
if test "$enable_opencv" = "yes"; then
    PKG_CHECK_MODULES(OPENCV, [opencv], [HAVE_OPENCV=1], [HAVE_OPENCV=0])
fi

# check dvs opencl path
ENABLE_DVS_CL_PATH=0
if test "$HAVE_OPENCV" -eq 1; then
    AC_LANG(C++)
    saved_CPPFLAGS="$CPPFLAGS"
    CPPFLAGS="$CPPFLAGS $OPENCV_CFLAGS"
    saved_LIBS="$LIBS"
    LIBS="$LIBS $OPENCV_LIBS"
    AC_COMPILE_IFELSE(
        [AC_LANG_PROGRAM(
            [[#include <opencv2/core.hpp>
              #include <opencv2/opencv.hpp>
            ]],
            [[cv::UMat frame0;
              cv::UMat frame1;
              cv::videostab::MotionEstimatorRansacL2 est;
              cv::videostab::KeypointBasedMotionEstimator kpest(&est);
              kpest.estimate(frame0, frame1);
            ]]
           )],
        [ENABLE_DVS_CL_PATH=1],
        [ENABLE_DVS_CL_PATH=0]
    )
    CPPFLAGS="$saved_CPPFLAGS"
    LIBS="$saved_LIBS"
fi

# check capi build
ENABLE_CAPI=0
if test "$enable_capi" = "yes"; then
    ENABLE_CAPI=1
fi

# check AIQ
ENABLE_IA_AIQ=0
USE_LOCAL_AIQ=0
if test "$enable_aiq" = "yes"; then
    ENABLE_IA_AIQ=1
    PKG_CHECK_MODULES(IA_AIQ, ia_imaging,
        PKG_CHECK_EXISTS(ia_imaging >= 2.7,
            AC_DEFINE(HAVE_AIQ_2_7, 1, [defined if module ia_imaging >= v2.0_007 is found])
        ),
        [USE_LOCAL_AIQ=1]
    )

    if test "$USE_LOCAL_AIQ" -eq 1; then
        #check HAVE_AIQ_2_7
        AC_CACHE_CHECK([for ext/ia_imaging >=v2.0_007],
            ac_cv_have_aiq_2_7, [
            saved_CPPFLAGS="$CPPFLAGS"
            CPPFLAGS="$CPPFLAGS -I./ext/ia_imaging/include"
            saved_LIBS="$LIBS"
            LIBS="$LIBS"
            AC_COMPILE_IFELSE(
                [AC_LANG_PROGRAM(
                    [[#include <stdint.h>
                      #include <stdio.h>
                      #include "ia_aiq_types.h"
                    ]],
                    [[ia_aiq_ae_results ae_result;
                      ae_result.flashes = NULL;
                    ]]
                   )],
                [ac_cv_have_aiq_2_7="yes"],
                [ac_cv_have_aiq_2_7="no"]
            )
            CPPFLAGS="$saved_CPPFLAGS"
            LIBS="$saved_LIBS"
        ])

        if test "$ac_cv_have_aiq_2_7" = "yes"; then
            AC_DEFINE(HAVE_AIQ_2_7, 1, [defined if ia_imaging >= v2.0_007 is found])
        fi
    fi

    if test "$USE_LOCAL_AIQ" -eq 0; then
        IA_IMAGING_CFLAGS="$IA_AIQ_CFLAGS"
        IA_IMAGING_LIBS="$IA_AIQ_LIBS"
    else
        IA_IMAGING_CFLAGS="-I\$(top_srcdir)/ext/ia_imaging/include"
        IA_IMAGING_LIBS="-L\$(top_srcdir)/ext/ia_imaging/lib -lia_aiq -lia_isp_2_2 -lia_cmc_parser -lia_mkn -lia_nvm -lia_exc -lia_log"
    fi
    AC_SUBST(IA_IMAGING_CFLAGS)
    AC_SUBST(IA_IMAGING_LIBS)
    LIBS="$LIBS $IA_IMAGING_LIBS"
fi

# check 3a lib build
ENABLE_3ALIB=0
if test "$enable_3alib" = "yes"; then
    ENABLE_3ALIB=1
fi
AM_CONDITIONAL([ENABLE_3ALIB], [test "$ENABLE_3ALIB" -eq 1])

# check smart analysis lib build
ENABLE_SMART_LIB=0
if test "$enable_smartlib" = "yes"; then
    ENABLE_SMART_LIB=1
fi
AM_CONDITIONAL([ENABLE_SMART_LIB], [test "$ENABLE_SMART_LIB" -eq 1])

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


# build gstreamer plugin
GST_API_VERSION=1.0
GST_VERSION_REQUIRED=1.2.3
ENABLE_GST=0
if test "$enable_gst" = "yes"; then
    ENABLE_GST=1
    PKG_CHECK_MODULES([GST], [gstreamer-$GST_API_VERSION >= $GST_VERSION_REQUIRED])
    PKG_CHECK_MODULES([GST_ALLOCATOR], [gstreamer-allocators-$GST_API_VERSION >= $GST_VERSION_REQUIRED])
    PKG_CHECK_MODULES([GST_VIDEO], [gstreamer-video-$GST_API_VERSION >= $GST_VERSION_REQUIRED])
fi
AM_CONDITIONAL([ENABLE_GST], [test "$ENABLE_GST" -eq 1])

dnl set XCAM_CFLAGS and XCAM_CXXFLAGS
XCAM_CFLAGS=" -fPIC -DSTDC99 -W -Wall -D_REENTRANT -Wformat -Wformat-security -fstack-protector"
if test "$enable_debug" = "yes"; then
    XCAM_CFLAGS="$XCAM_CFLAGS -g -DDEBUG"
fi
XCAM_CXXFLAGS="$XCAM_CFLAGS -std=c++0x"
AC_SUBST(XCAM_CFLAGS)
AC_SUBST(XCAM_CXXFLAGS)

PTHREAD_LDFLAGS="$PTHREAD_LDFLAGS -pthread"
AC_SUBST(PTHREAD_LDFLAGS)

# define macor in config.h
AC_DEFINE_UNQUOTED([ENABLE_PROFILING], $ENABLE_PROFILING,
    [enable profiling])

AC_DEFINE_UNQUOTED([HAVE_LIBDRM], $HAVE_LIBDRM,
    [have libdrm])
AM_CONDITIONAL([HAVE_LIBDRM], [test "$HAVE_LIBDRM" -eq 1])

AC_DEFINE_UNQUOTED([HAVE_LIBCL], $HAVE_LIBCL,
    [have libcl])
AM_CONDITIONAL([HAVE_LIBCL], [test "$HAVE_LIBCL" -eq 1])

AC_DEFINE_UNQUOTED([HAVE_OPENCV], $HAVE_OPENCV,
    [have opencv])
AM_CONDITIONAL([HAVE_OPENCV], [test "$HAVE_OPENCV" -eq 1])

AC_DEFINE_UNQUOTED([ENABLE_DVS_CL_PATH], $ENABLE_DVS_CL_PATH,
    [enable dvs cl path])
AM_CONDITIONAL([ENABLE_DVS_CL_PATH], [test "$ENABLE_DVS_CL_PATH" -eq 1])

AC_DEFINE_UNQUOTED([ENABLE_CAPI], $ENABLE_CAPI,
    [enable capi build])
AM_CONDITIONAL([ENABLE_CAPI], [test "$ENABLE_CAPI" -eq 1])

#atomisp
AM_CONDITIONAL([USE_LOCAL_ATOMISP], [test "$USE_LOCAL_ATOMISP" -eq 1])

# aiq (ia_imaging)
AC_DEFINE_UNQUOTED([HAVE_IA_AIQ], $ENABLE_IA_AIQ,
    [have aiq binary])
AM_CONDITIONAL([ENABLE_IA_AIQ], [test "$ENABLE_IA_AIQ" -eq 1])
AM_CONDITIONAL([USE_LOCAL_AIQ], [test "$USE_LOCAL_AIQ" -eq 1])

AC_CONFIG_FILES([Makefile
                 clx_kernel/Makefile
                 xcore/Makefile
                 modules/Makefile
                 modules/isp/Makefile
                 modules/ocl/Makefile
                 wrapper/Makefile
                 wrapper/gstreamer/Makefile
                 wrapper/gstreamer/interface/Makefile
                 plugins/Makefile
                 plugins/3a/hybrid/Makefile
                 plugins/3a/aiq/Makefile
                 plugins/3a/Makefile
                 plugins/smart/Makefile
                 plugins/smart/dvs/Makefile
                 plugins/smart/dvs/libdvs/Makefile
                 plugins/smart/sample/Makefile
                 capi/Makefile
                 tests/Makefile
                 pkgconfig/Makefile
                 pkgconfig/libxcam.pc
                ])

AC_OUTPUT

if test "$USE_LOCAL_AIQ" -eq 1; then use_local_aiq="yes"; else  use_local_aiq="no"; fi
if test "$USE_LOCAL_ATOMISP" -eq 1; then use_local_atomisp="yes"; else  use_local_atomisp="no"; fi
if test "$HAVE_LIBCL" -eq 1; then have_libcl="yes"; else  have_libcl="no"; fi
if test "$HAVE_OPENCV" -eq 1; then have_opencv="yes"; else have_opencv="no"; fi

echo "
     libxcam configuration summary
     version                    : $XCAM_VERSION
     enable debug               : $enable_debug
     enable profiling           : $enable_profiling
     build GStreamer plugin     : $enable_gst
     build aiq analyzer         : $enable_aiq
     use local aiq              : $use_local_aiq
     use local atomisp          : $use_local_atomisp
     have opencl lib            : $have_libcl
     have opencv lib            : $have_opencv
     enable 3a lib              : $enable_3alib
     enable smart analysis lib  : $enable_smartlib
     enable libxcam-capi lib    : $enable_capi
"
