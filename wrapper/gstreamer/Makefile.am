if ENABLE_IA_AIQ
SUBDIRS = interface
endif

plugin_LTLIBRARIES = libgstxcamsrc.la

if HAVE_LIBCL
plugin_LTLIBRARIES += libgstxcamfilter.la
endif

XCAM_GST_CXXFLAGS = \
    $(XCAM_CXXFLAGS)                  \
    $(GST_CFLAGS)                     \
    -I$(top_srcdir)/xcore             \
    -I$(top_srcdir)/modules           \
    -I$(top_srcdir)/wrapper/gstreamer \
    $(NULL)

XCAM_GST_LIBS = \
    $(GST_LIBS)                           \
    $(GST_VIDEO_LIBS)                     \
    $(GST_ALLOCATOR_LIBS)                 \
    $(top_builddir)/xcore/libxcam_core.la \
    $(NULL)

if HAVE_LIBCL
XCAM_GST_LIBS += $(top_builddir)/modules/ocl/libxcam_ocl.la
endif

if HAVE_LIBDRM
XCAM_GST_CXXFLAGS += $(LIBDRM_CFLAGS)
XCAM_GST_LIBS += $(LIBDRM_LIBS)
endif

# Note: plugindir is set in configure
plugindir="$(libdir)/gstreamer-1.0"

# sources used to compile this plug-in
libgstxcamsrc_la_SOURCES = \
    gstxcambuffermeta.cpp \
    gstxcambufferpool.cpp \
    main_dev_manager.cpp  \
    gstxcamsrc.cpp        \
    $(NULL)

# compiler and linker flags used to compile this plugin, set in configure.ac
XCAM_SRC_CXXFLAGS = \
    $(XCAM_GST_CXXFLAGS) \
    $(NULL)

XCAM_SRC_LIBS = \
    $(XCAM_GST_LIBS) \
    $(NULL)

if USE_LOCAL_ATOMISP
XCAM_SRC_CXXFLAGS += -I$(top_srcdir)/ext/atomisp
endif

if ENABLE_IA_AIQ
XCAM_SRC_CXXFLAGS += -I$(top_srcdir)/wrapper/gstreamer/interface
XCAM_SRC_LIBS += \
    $(top_builddir)/modules/isp/libxcam_isp.la                          \
    $(top_builddir)/wrapper/gstreamer/interface/libgstxcaminterface.la \
    $(NULL)
endif

libgstxcamsrc_la_CXXFLAGS = \
    $(XCAM_SRC_CXXFLAGS) \
    $(NULL)

libgstxcamsrc_la_LIBADD = \
    $(XCAM_SRC_LIBS) \
    $(NULL)

libgstxcamsrc_la_LDFLAGS = \
    -module -avoid-version \
    $(NULL)

libgstxcamsrc_la_LIBTOOLFLAGS = --tag=disable-static

if HAVE_LIBCL
libgstxcamfilter_la_SOURCES = \
    gstxcambuffermeta.cpp \
    main_pipe_manager.cpp \
    gstxcamfilter.cpp     \
    $(NULL)

libgstxcamfilter_la_CXXFLAGS = \
    $(XCAM_GST_CXXFLAGS) \
    $(NULL)

libgstxcamfilter_la_LIBADD = \
    $(XCAM_GST_LIBS) \
    $(NULL)

libgstxcamfilter_la_LDFLAGS = \
    -module -avoid-version \
    $(NULL)

libgstxcamfilter_la_LIBTOOLFLAGS = --tag=disable-static
endif

# headers we need but do not want installed
noinst_HEADERS = \
    gst_xcam_utils.h    \
    gstxcambufferpool.h \
    gstxcambuffermeta.h \
    main_dev_manager.h  \
    gstxcamsrc.h        \
    $(NULL)

if HAVE_LIBCL
noinst_HEADERS += \
    gstxcambuffermeta.h \
    main_pipe_manager.h \
    gstxcamfilter.h     \
    $(NULL)
endif
