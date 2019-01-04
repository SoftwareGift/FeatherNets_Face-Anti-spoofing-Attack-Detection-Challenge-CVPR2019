plugin_LTLIBRARIES = libxcam_3a_aiq.la

XCAM_AIQ_CXXFLAGS = \
    $(XCAM_CXXFLAGS)               \
    -I$(top_srcdir)/xcore          \
    -I$(top_srcdir)/modules/isp    \
    -I$(top_srcdir)/plugins/3a/aiq \
    $(NULL)

XCAM_AIQ_LIBS = \
    $(top_builddir)/xcore/libxcam_core.la      \
    $(top_builddir)/modules/isp/libxcam_isp.la \
    $(NULL)

if HAVE_LIBDRM
XCAM_AIQ_CXXFLAGS += $(LIBDRM_CFLAGS)
XCAM_AIQ_LIBS += $(LIBDRM_LIBS)
endif

if USE_LOCAL_ATOMISP
XCAM_AIQ_CXXFLAGS += \
    -I$(top_srcdir)/ext/atomisp \
    $(NULL)
endif

plugindir="$(libdir)/xcam/plugins/3a"

libxcam_3a_aiq_la_SOURCES = \
    aiq_wrapper.cpp \
    $(NULL)

libxcam_3a_aiq_la_CXXFLAGS = \
    $(XCAM_AIQ_CXXFLAGS) \
    $(NULL)

libxcam_3a_aiq_la_LIBADD = \
    $(XCAM_AIQ_LIBS) \
    $(NULL)

libxcam_3a_aiq_la_LDFLAGS = \
    -module -avoid-version \
    $(NULL)

libxcam_3a_aiq_la_LIBTOOLFLAGS = --tag=disable-static
