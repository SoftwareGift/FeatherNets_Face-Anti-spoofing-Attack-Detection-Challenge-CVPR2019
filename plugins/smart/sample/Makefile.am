noinst_LTLIBRARIES = libxcam_sample_smart.la

XCAM_SAMPLE_CXXFLAGS = \
    $(XCAM_CXXFLAGS)              \
    -I$(top_srcdir)/xcore         \
    -I$(top_srcdir)/modules/isp   \
    -I$(top_srcdir)/plugins/smart \
    $(NULL)

XCAM_SAMPLE_LIBS = \
    $(top_builddir)/xcore/libxcam_core.la \
    $(NULL)

if USE_LOCAL_ATOMISP
XCAM_SAMPLE_CXXFLAGS += \
    -I$(top_srcdir)/ext/atomisp \
    $(NULL)
endif

plugindir="$(libdir)/xcam/plugins/smart"

libxcam_sample_smart_la_SOURCES = \
    sample_smart_analysis.cpp \
    $(NULL)

libxcam_sample_smart_la_CXXFLAGS = \
    $(XCAM_SAMPLE_CXXFLAGS) \
    $(NULL)

libxcam_sample_smart_la_LIBADD = \
    $(XCAM_SAMPLE_LIBS) \
    $(NULL)

libxcam_sample_smart_la_LDFLAGS = \
    -module -avoid-version \
    $(NULL)

libxcam_sample_smart_la_LIBTOOLFLAGS = --tag=disable-static
