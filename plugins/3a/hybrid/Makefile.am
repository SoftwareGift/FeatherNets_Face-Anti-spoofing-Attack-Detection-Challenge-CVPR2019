plugin_LTLIBRARIES = libxcam_3a_hybrid.la

XCAMSRC_CXXFLAGS = $(XCAM_CXXFLAGS)
XCAMSRC_LIBS = \
    $(NULL)

plugindir="$(libdir)/xcam/plugins/3a"

libxcam_3a_hybrid_la_SOURCES = \
    sample.cpp        \
    $(NULL)

libxcam_3a_hybrid_la_CXXFLAGS = \
    $(GST_CFLAGS) $(XCAMSRC_CXXFLAGS)      \
    -I$(top_builddir)/xcore                \
    -I$(top_builddir)/plugins/3a/hybrid    \
    $(NULL)

libxcam_3a_hybrid_la_LIBADD = \
    $(XCAMSRC_LIBS)                        \
    $(top_builddir)/xcore/libxcam_core.la  \
    $(NULL)

libxcam_3a_hybrid_la_LDFLAGS = \
    -module -avoid-version                 \
    $(top_builddir)/xcore/libxcam_core.la  \
    $(PTHREAD_LDFLAGS)                     \
    $(NULL)

libxcam_3a_hybrid_la_LIBTOOLFLAGS = --tag=disable-static
