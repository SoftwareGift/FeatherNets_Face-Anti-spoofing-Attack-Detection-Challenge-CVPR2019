plugin_LTLIBRARIES = libxcam_sample_smart.la

XCAMSRC_CXXFLAGS = $(XCAM_CXXFLAGS)
XCAMSRC_LIBS = \
		   $(NULL)

plugindir="$(libdir)/xcam/smartlib"

libxcam_sample_smart_la_SOURCES =               \
			   sample_smart_analysis.cpp  \
			   $(NULL)

libxcam_sample_smart_la_CXXFLAGS = -I$(top_builddir)/xcore         \
			  $(GST_CFLAGS) $(XCAMSRC_CXXFLAGS)	\
			  -I$(top_builddir)/smartlib             \
			  $(NULL)


libxcam_sample_smart_la_LIBADD = $(XCAMSRC_LIBS)                   \
			  $(top_builddir)/xcore/libxcam_core.la \
			  $(NULL)

libxcam_sample_smart_la_LDFLAGS =                                  \
			  -module  -avoid-version               \
			  $(top_builddir)/xcore/libxcam_core.la \
			  $(PTHREAD_LDFLAGS)                    \
			  $(NULL)

libxcam_sample_smart_la_LIBTOOLFLAGS = --tag=disable-static
