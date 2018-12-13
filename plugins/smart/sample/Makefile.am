noinst_LTLIBRARIES = \
    $(NULL)

if ENABLE_IA_AIQ
noinst_LTLIBRARIES += \
    libxcam_sample_smart.la \
    $(NULL)
endif

XCAMSMART_CXXFLAGS = $(XCAM_CXXFLAGS)
XCAMSMART_LIBS = \
    $(NULL)

if USE_LOCAL_ATOMISP
XCAMSMART_CXXFLAGS += \
    -I$(top_srcdir)/ext/atomisp  \
    $(NULL)
endif

plugindir="$(libdir)/xcam/plugins/smart"

if ENABLE_IA_AIQ
libxcam_sample_smart_la_SOURCES = \
    sample_smart_analysis.cpp    \
    $(NULL)

libxcam_sample_smart_la_CXXFLAGS = \
    $(GST_CFLAGS) $(XCAMSMART_CXXFLAGS)  \
    -I$(top_srcdir)/xcore                \
    -I$(top_srcdir)/modules/isp          \
    -I$(top_srcdir)/plugins/smart        \
    $(NULL)

libxcam_sample_smart_la_LIBADD = \
    $(XCAMSMART_LIBS)                      \
    $(top_builddir)/xcore/libxcam_core.la  \
    $(NULL)

libxcam_sample_smart_la_LDFLAGS = \
    -module -avoid-version                 \
    $(top_builddir)/xcore/libxcam_core.la  \
    $(NULL)
endif

libxcam_sample_smart_la_LIBTOOLFLAGS = --tag=disable-static
