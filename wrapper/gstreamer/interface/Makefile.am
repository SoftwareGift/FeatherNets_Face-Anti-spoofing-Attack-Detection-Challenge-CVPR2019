lib_LTLIBRARIES = libgstxcaminterface.la

libgstxcaminterface_la_SOURCES = gstxcaminterface.c

libgstxcaminterface_la_CFLAGS =              \
				 $(GST_CFLAGS)               \
				 -I$(top_srcdir)/xcore       \
				 -I$(top_srcdir)/xcore/base  \
				 $(XCAM_CFLAGS)              \
				 $(NULL)

libgstxcaminterface_la_LIBADD = \
				 $(GST_LIBS) \
				 $(NULL)

libgstxcaminterfaceincludedir = \
	$(includedir)/gstreamer-1.0/gst

libgstxcaminterfaceinclude_HEADERS = \
	gstxcaminterface.h

