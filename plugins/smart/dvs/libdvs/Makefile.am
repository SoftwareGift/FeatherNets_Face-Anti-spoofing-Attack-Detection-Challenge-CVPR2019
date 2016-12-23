noinst_LTLIBRARIES =  \
	libxcam_dvs.la    \
	$(NULL)

XCAM_DVS_LIB_CXXFLAGS = \
	$(XCAM_CXXFLAGS) \
	$(OPENCV_CFLAGS) \
	$(NULL)
                             
XCAM_DVS_LIBS =      \
	$(OPENCV_LIBS)   \
	$(NULL)


libxcam_dvs_la_SOURCES = \
	libdvs.cpp           \
	stabilizer.cpp       \
	$(NULL)

libxcam_dvs_la_CXXFLAGS =                  \
	$(GST_CFLAGS) $(XCAM_DVS_LIB_CXXFLAGS) \
	-I$(top_builddir)/xcore                \
	-I$(top_builddir)/modules/ocl          \
	$(NULL)

libxcam_dvs_la_LIBADD =                    \
    $(XCAM_DVS_LIBS)                       \
    $(top_builddir)/xcore/libxcam_core.la  \
    $(top_builddir)/modules/ocl/libxcam_ocl.la \
    $(NULL)

libxcam_dvs_la_LDFLAGS =                   \
    -module -avoid-version                 \
    $(top_builddir)/xcore/libxcam_core.la  \
    $(PTHREAD_LDFLAGS)                     \
    $(NULL)

libxcam_dvs_la_LIBTOOLFLAGS = --tag=disable-static
