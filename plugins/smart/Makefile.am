noinst_LTLIBRARIES = libxcam_sample_smart.la

XCAMSMART_CXXFLAGS = $(XCAM_CXXFLAGS)
XCAMSMART_LIBS = \
    $(NULL)

if USE_LOCAL_ATOMISP
XCAMSMART_CXXFLAGS += \
    -I$(top_srcdir)/ext/atomisp  \
    $(NULL)
endif

plugindir="$(libdir)/xcam/plugins/smart"

libxcam_sample_smart_la_SOURCES = \
    sample_smart_analysis.cpp    \
    $(NULL)

libxcam_sample_smart_la_CXXFLAGS = \
    $(GST_CFLAGS) $(XCAMSMART_CXXFLAGS)   \
    -I$(top_builddir)/xcore               \
    -I$(top_builddir)/modules/isp         \
    -I$(top_builddir)/plugins/smart       \
    $(NULL)

libxcam_sample_smart_la_LIBADD = \
    $(XCAMSMART_LIBS)                      \
    $(top_builddir)/xcore/libxcam_core.la  \
    $(NULL)

libxcam_sample_smart_la_LDFLAGS = \
    -module -avoid-version                 \
    $(top_builddir)/xcore/libxcam_core.la  \
    $(PTHREAD_LDFLAGS)                     \
    $(NULL)

libxcam_sample_smart_la_LIBTOOLFLAGS = --tag=disable-static
