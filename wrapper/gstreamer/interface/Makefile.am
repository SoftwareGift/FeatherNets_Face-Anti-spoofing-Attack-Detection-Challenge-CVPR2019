lib_LTLIBRARIES = libgstxcaminterface.la

libgstxcaminterface_la_SOURCES = gstxcaminterface.c

libgstxcaminterface_la_CFLAGS = \
    $(XCAM_CFLAGS)             \
    $(GST_CFLAGS)              \
    -I$(top_srcdir)/xcore      \
    -I$(top_srcdir)/xcore/base \
    $(NULL)

libgstxcaminterface_la_LIBADD = \
    $(GST_LIBS)                           \
    $(top_builddir)/xcore/libxcam_core.la \
    $(NULL)

libgstxcaminterfaceincludedir = \
    $(includedir)/gstreamer-1.0/gst \
    $(NULL)

libgstxcaminterfaceinclude_HEADERS = \
    gstxcaminterface.h \
    $(NULL)
