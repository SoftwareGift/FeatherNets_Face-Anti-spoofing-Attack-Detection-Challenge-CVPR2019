noinst_LIBRARIES =  \
	libxcam_dvs.a    \
	$(NULL)

XCAM_DVS_LIB_CXXFLAGS = \
	$(XCAM_CXXFLAGS) \
	$(OPENCV_CFLAGS) \
	$(NULL)
                             
XCAM_DVS_LIBS =      \
	$(OPENCV_LIBS)   \
	$(NULL)

libxcam_dvs_a_SOURCES = \
	libdvs.cpp           \
	stabilizer.cpp       \
	$(NULL)

libxcam_dvs_a_CXXFLAGS =                  \
	$(GST_CFLAGS) $(XCAM_DVS_LIB_CXXFLAGS) \
	-I$(top_builddir)/xcore                \
	-I$(top_builddir)/modules/ocl          \
	$(NULL)

libxcam_dvs_a_LIBADD =                    \
    $(top_builddir)/xcore/libxcam_core.la  \
    $(top_builddir)/modules/ocl/libxcam_ocl.la \
    $(NULL)

noinst_PROGRAMS = \
    test_image_stabilization \
    $(NULL)

test_image_stabilization_SOURCES = \
    test-image-stabilization.cpp \
    $(NULL)

test_image_stabilization_CXXFLAGS = \
    $(XCAM_CXXFLAGS) \
    -I$(top_builddir)/xcore \
    $(NULL)

test_image_stabilization_LDADD = \
    libxcam_dvs.a \
    $(OPENCV_LIBS) \
    $(NULL)

