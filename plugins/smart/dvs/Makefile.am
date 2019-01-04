SUBDIRS = libdvs

noinst_LTLIBRARIES = libxcam_plugin_dvs.la

XCAM_PLUGIN_DVS_CXXFLAGS = \
    $(XCAM_CXXFLAGS)                         \
    -I$(top_srcdir)/xcore                    \
    -I$(top_srcdir)/modules                  \
    -I$(top_srcdir)/plugins/smart/dvs/libdvs \
    $(NULL)

XCAM_PLUGIN_DVS_LIBS = \
    $(OPENCV_LIBS)                                         \
    $(top_builddir)/xcore/libxcam_core.la                  \
    $(top_builddir)/modules/ocl/libxcam_ocl.la             \
    $(top_builddir)/plugins/smart/dvs/libdvs/libxcam_dvs.a \
    $(NULL)

if HAVE_LIBDRM
XCAM_PLUGIN_DVS_CXXFLAGS += $(LIBDRM_CFLAGS)
XCAM_PLUGIN_DVS_LIBS += $(LIBDRM_LIBS)
endif

plugindir="$(libdir)/xcam/plugins/smart"

libxcam_plugin_dvs_la_SOURCES = \
    xcam_plugin_dvs.cpp \
    $(NULL)

libxcam_plugin_dvs_la_CXXFLAGS = \
    $(XCAM_PLUGIN_DVS_CXXFLAGS) \
    $(NULL)

libxcam_plugin_dvs_la_LIBADD = \
    $(XCAM_PLUGIN_DVS_LIBS) \
    $(NULL)

libxcam_plugin_dvs_la_LDFLAGS = \
    -module -avoid-version \
    $(NULL)

libxcam_plugin_dvs_la_LIBTOOLFLAGS = --tag=disable-static
