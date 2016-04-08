SUBDIRS = interface

plugin_LTLIBRARIES = libgstxcamsrc.la

PTHREAD_LDFLAGS = -pthread

XCAMSRC_CXXFLAGS = $(XCAM_CXXFLAGS)
XCAMSRC_LIBS = \
		   $(NULL)

if HAVE_LIBDRM
XCAMSRC_CXXFLAGS += $(LIBDRM_CFLAGS)
XCAMSRC_LIBS += $(LIBDRM_LIBS)
endif

if USE_LOCAL_ATOMISP
XCAMSRC_CXXFLAGS +=        \
    -I$(top_srcdir)/ext/atomisp \
    $(NULL)
endif

# Note: plugindir is set in configure
plugindir="$(libdir)/gstreamer-1.0"

# sources used to compile this plug-in
libgstxcamsrc_la_SOURCES =                  \
			   gstxcambuffermeta.cpp        \
			   gstxcamsrc.cpp               \
			   gstxcambufferpool.cpp        \
			   main_dev_manager.cpp         \
			   $(NULL)

# compiler and linker flags used to compile this plugin, set in configure.ac
libgstxcamsrc_la_CXXFLAGS = -I$(top_builddir)/xcore -I$(top_builddir)/xcore/base   \
			   $(GST_CFLAGS) $(XCAMSRC_CXXFLAGS)	\
			  -I$(top_builddir)/wrapper/gstreamer   \
			  -I$(top_builddir)/wrapper/gstreamer/interface \
			  $(NULL)


libgstxcamsrc_la_LIBADD = $(XCAMSRC_LIBS)          \
			  $(top_builddir)/wrapper/gstreamer/interface/libgstxcaminterface.la	    \
			  $(top_builddir)/xcore/libxcam_core.la \
			  $(GST_ALLOCATOR_LIBS)                 \
			  $(GST_VIDEO_LIBS)                     \
			  $(GST_LIBS)

libgstxcamsrc_la_LDFLAGS =                          \
			  -module  -avoid-version               \
			  $(top_builddir)/xcore/libxcam_core.la \
			  $(PTHREAD_LDFLAGS)                    \
			  $(NULL)

libgstxcamsrc_la_LIBTOOLFLAGS = --tag=disable-static

# headers we need but don't want installed
noinst_HEADERS =                          \
			   gstxcambufferpool.h        \
			   gstxcamsrc.h               \
			   gstxcambuffermeta.h        \
			   main_dev_manager.h         \
			   $(NULL)
