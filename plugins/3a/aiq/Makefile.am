plugin_LTLIBRARIES = libxcam_3a_aiq.la

XCAMAIQ_CXXFLAGS = $(XCAM_CXXFLAGS)
XCAMAIQ_LIBS = \
    $(NULL)

if HAVE_LIBDRM
XCAMAIQ_CXXFLAGS += $(LIBDRM_CFLAGS)
XCAMAIQ_LIBS += $(LIBDRM_LIBS)
endif

if USE_LOCAL_ATOMISP
XCAMAIQ_CXXFLAGS += \
    -I$(top_srcdir)/ext/atomisp  \
    $(NULL)
endif

plugindir="$(libdir)/xcam/plugins/3a"

libxcam_3a_aiq_la_SOURCES = \
    aiq_wrapper.cpp  \
    $(NULL)

libxcam_3a_aiq_la_CXXFLAGS = \
    $(GST_CFLAGS) $(XCAMAIQ_CXXFLAGS)      \
    -I$(top_builddir)/xcore                \
    -I$(top_builddir)/modules/isp          \
    -I$(top_builddir)/plugins/3a/aiq       \
    $(NULL)

libxcam_3a_aiq_la_LIBADD = \
    $(XCAMAIQ_LIBS)                             \
    $(top_builddir)/modules/isp/libxcam_isp.la  \
    $(top_builddir)/xcore/libxcam_core.la       \
    $(NULL)

libxcam_3a_aiq_la_LDFLAGS = \
    -module -avoid-version                 \
    $(top_builddir)/xcore/libxcam_core.la  \
    $(PTHREAD_LDFLAGS)                     \
    $(NULL)

libxcam_3a_aiq_la_LIBTOOLFLAGS = --tag=disable-static
