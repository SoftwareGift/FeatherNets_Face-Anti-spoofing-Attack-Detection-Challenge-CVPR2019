noinst_LTLIBRARIES = libxcam_ocv.la

xcam_ocv_sources = \
    cv_utils.cpp                 \
    cv_image_process_helper.cpp  \
    cv_image_sharp.cpp           \
    cv_edgetaper.cpp             \
    cv_wiener_filter.cpp         \
    cv_image_deblurring.cpp      \
    cv_feature_match.cpp         \
    cv_feature_match_cluster.cpp \
    cv_capi_feature_match.cpp    \
    $(NULL)

libxcam_ocv_la_SOURCES = \
    $(xcam_ocv_sources) \
    $(NULL)

libxcam_ocv_la_CXXFLAGS = \
    $(XCAM_CXXFLAGS)        \
    $(OPENCV_CFLAGS)        \
    -I$(top_srcdir)/xcore   \
    -I$(top_srcdir)/modules \
    $(NULL)

libxcam_ocv_la_LIBADD = \
    $(OPENCV_LIBS)                        \
    $(top_builddir)/xcore/libxcam_core.la \
    $(NULL)