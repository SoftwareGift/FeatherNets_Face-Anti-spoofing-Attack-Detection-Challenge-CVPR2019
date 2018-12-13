if ENABLE_IA_AIQ
SUBDIRS = interface
endif

plugin_LTLIBRARIES = \
    libgstxcamsrc.la \
    $(NULL)

if HAVE_LIBCL
plugin_LTLIBRARIES += \
    libgstxcamfilter.la \
    $(NULL)
endif

XCORE_DIR = $(top_srcdir)/xcore
MODULES_DIR = $(top_srcdir)/modules

XCORE_LA = $(top_builddir)/xcore/libxcam_core.la

if ENABLE_IA_AIQ
XCAM_INTERFACE_DIR = -I$(top_srcdir)/wrapper/gstreamer/interface
XCAM_INTERFACE_LA = $(top_builddir)/wrapper/gstreamer/interface/libgstxcaminterface.la
else
XCAM_INTERFACE_DIR =
XCAM_INTERFACE_LA =
endif

XCAMGST_CXXFLAGS = $(XCAM_CXXFLAGS)
XCAMGST_LIBS = \
    $(NULL)

if HAVE_LIBDRM
XCAMGST_CXXFLAGS += $(LIBDRM_CFLAGS)
XCAMGST_LIBS += $(LIBDRM_LIBS)
endif

if USE_LOCAL_ATOMISP
XCAMGST_CXXFLAGS += -I$(top_srcdir)/ext/atomisp
endif

XCAMGST_CXXFLAGS += \
    -I$(XCORE_DIR)   \
    -I$(MODULES_DIR) \
    $(NULL)

# Note: plugindir is set in configure
plugindir="$(libdir)/gstreamer-1.0"

# sources used to compile this plug-in
libgstxcamsrc_la_SOURCES = \
    gstxcambuffermeta.cpp  \
    gstxcambufferpool.cpp  \
    main_dev_manager.cpp   \
    gstxcamsrc.cpp         \
    $(NULL)

# compiler and linker flags used to compile this plugin, set in configure.ac
libgstxcamsrc_la_CXXFLAGS = \
    $(GST_CFLAGS) $(XCAMGST_CXXFLAGS)  \
    -I$(top_srcdir)/wrapper/gstreamer  \
    $(XCAM_INTERFACE_DIR)              \
    $(NULL)

libgstxcamsrc_la_LIBADD = $(XCAMGST_LIBS) \
    $(XCAM_INTERFACE_LA)              \
    $(XCORE_LA) $(GST_ALLOCATOR_LIBS) \
    $(GST_VIDEO_LIBS) $(GST_LIBS)     \
    $(NULL)

libgstxcamsrc_la_LDFLAGS = \
    -module -avoid-version $(XCORE_LA) \
    $(NULL)

libgstxcamsrc_la_LIBTOOLFLAGS = --tag=disable-static

if ENABLE_IA_AIQ
ISP_LA = $(top_builddir)/modules/isp/libxcam_isp.la
libgstxcamsrc_la_LIBADD += $(ISP_LA)
libgstxcamsrc_la_LDFLAGS += $(ISP_LA)
endif

if HAVE_LIBCL
OCL_LA = $(top_builddir)/modules/ocl/libxcam_ocl.la

libgstxcamsrc_la_LIBADD += $(OCL_LA)
libgstxcamsrc_la_LDFLAGS += $(OCL_LA)

libgstxcamfilter_la_SOURCES = \
    gstxcambuffermeta.cpp  \
    main_pipe_manager.cpp  \
    gstxcamfilter.cpp      \
    $(NULL)

libgstxcamfilter_la_CXXFLAGS = \
    $(GST_CFLAGS) $(XCAMGST_CXXFLAGS) \
    -I$(top_srcdir)/wrapper/gstreamer \
    $(NULL)

libgstxcamfilter_la_LIBADD = \
    $(XCAMGST_LIBS)        \
    $(XCORE_LA) $(OCL_LA)  \
    $(GST_ALLOCATOR_LIBS)  \
    $(GST_VIDEO_LIBS)      \
    $(GST_LIBS)            \
    $(NULL)

libgstxcamfilter_la_LDFLAGS = \
    -module -avoid-version    \
    $(XCORE_LA) $(OCL_LA)     \
    $(NULL)

libgstxcamfilter_la_LIBTOOLFLAGS = --tag=disable-static
endif

# headers we need but don't want installed
noinst_HEADERS = \
    gst_xcam_utils.h  \
    $(NULL)

if ENABLE_IA_AIQ
noinst_HEADERS += \
    gstxcambufferpool.h  \
    gstxcambuffermeta.h  \
    main_dev_manager.h   \
    gstxcamsrc.h         \
    $(NULL)
endif

if HAVE_LIBCL
noinst_HEADERS += \
    gstxcambuffermeta.h  \
    main_pipe_manager.h  \
    gstxcamfilter.h      \
    $(NULL)
endif
