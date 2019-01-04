plugin_LTLIBRARIES = libxcam_3a_hybrid.la

XCAM_HYBRID_CXXFLAGS = \
    $(XCAM_CXXFLAGS)                  \
    -I$(top_srcdir)/xcore             \
    -I$(top_srcdir)/plugins/3a/hybrid \
    $(NULL)

XCAM_HYBRID_LIBS = \
    $(top_builddir)/xcore/libxcam_core.la \
    $(NULL)

plugindir="$(libdir)/xcam/plugins/3a"

libxcam_3a_hybrid_la_SOURCES = \
    sample.cpp \
    $(NULL)

libxcam_3a_hybrid_la_CXXFLAGS = \
    $(XCAM_HYBRID_CXXFLAGS) \
    $(NULL)

libxcam_3a_hybrid_la_LIBADD = \
    $(XCAM_HYBRID_LIBS) \
    $(NULL)

libxcam_3a_hybrid_la_LDFLAGS = \
    -module -avoid-version \
    $(NULL)

libxcam_3a_hybrid_la_LIBTOOLFLAGS = --tag=disable-static
