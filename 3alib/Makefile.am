plugin_LTLIBRARIES = libxcam_3a_aiq.la

PTHREAD_LDFLAGS = -pthread

XCAMSRC_CXXFLAGS = $(XCAM_CXXFLAGS)
XCAMSRC_LIBS = \
		   $(NULL)

if HAVE_LIBDRM
XCAMSRC_CXXFLAGS += $(LIBDRM_CFLAGS)
XCAMSRC_LIBS += $(LIBDRM_LIBS)
endif

plugindir="$(libdir)/xcam"

libxcam_3a_aiq_la_SOURCES =                  \
			   aiq_wrapper.cpp           \
			   aiq3a_utils.cpp           \
			   $(NULL)

libxcam_3a_aiq_la_CXXFLAGS = -I$(top_builddir)/xcore    \
			  -I$(top_builddir)/xcore/base	        \
			  $(GST_CFLAGS) $(XCAMSRC_CXXFLAGS)	\
			  -I$(top_builddir)/3alib               \
			  $(NULL)


libxcam_3a_aiq_la_LIBADD = $(XCAMSRC_LIBS)           \
			  $(top_builddir)/xcore/libxcam_core.la \
			  $(NULL)

libxcam_3a_aiq_la_LDFLAGS =                          \
			  -module  -avoid-version               \
			  $(top_builddir)/xcore/libxcam_core.la \
			  $(PTHREAD_LDFLAGS)                    \
			  $(NULL)

libxcam_3a_aiq_la_LIBTOOLFLAGS = --tag=disable-static

noinst_HEADERS = aiq3a_utils.h     \
			   $(NULL)
