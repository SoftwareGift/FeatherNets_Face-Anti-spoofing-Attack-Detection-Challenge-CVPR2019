noinst_PROGRAMS = \
	test-device-manager  \
	$(NULL)

if ENABLE_IA_AIQ
noinst_PROGRAMS += \
	test-poll-thread     \
	$(NULL)
endif

if HAVE_LIBCL
noinst_PROGRAMS += \
	test-cl-image        \
	test-binary-kernel   \
	test-pipe-manager    \
	test-image-blend     \
	test-image-stitching \
	test-video-stabilization \
	$(NULL)

if HAVE_OPENCV
noinst_PROGRAMS += \
	test-image-deblurring \
	$(NULL)
endif
endif

XCORE_DIR = $(top_srcdir)/xcore
MODULES_DIR = $(top_srcdir)/modules

XCORE_LA = $(top_builddir)/xcore/libxcam_core.la

tests_cxxflags = $(XCAM_CXXFLAGS)
if HAVE_LIBDRM
tests_cxxflags += $(LIBDRM_CFLAGS) $(LIBDRM_LIBS)
endif

if USE_LOCAL_ATOMISP
tests_cxxflags += -I$(top_srcdir)/ext/atomisp
endif

tests_cxxflags += \
	-I$(XCORE_DIR)   \
	-I$(MODULES_DIR) \
	$(NULL)

test_device_manager_SOURCES = test-device-manager.cpp
test_device_manager_CXXFLAGS = $(tests_cxxflags)
test_device_manager_LDADD = $(XCORE_LA)

if ENABLE_IA_AIQ
ISP_LA = $(top_builddir)/modules/isp/libxcam_isp.la

test_device_manager_LDADD += $(ISP_LA)

test_poll_thread_SOURCES = test-poll-thread.cpp
test_poll_thread_CXXFLAGS = $(tests_cxxflags)
test_poll_thread_LDADD = \
	$(XCORE_LA) $(ISP_LA)  \
	$(NULL)
endif

if HAVE_LIBCL
OCL_LA = $(top_builddir)/modules/ocl/libxcam_ocl.la

test_device_manager_LDADD += $(OCL_LA)

test_cl_image_SOURCES = test-cl-image.cpp
test_cl_image_CXXFLAGS = $(tests_cxxflags)
test_cl_image_LDADD = \
	$(XCORE_LA) $(OCL_LA)  \
	$(NULL)

test_binary_kernel_SOURCES = test-binary-kernel.cpp
test_binary_kernel_CXXFLAGS = $(tests_cxxflags)
test_binary_kernel_LDADD = \
	$(XCORE_LA) $(OCL_LA)  \
	$(NULL)

test_pipe_manager_SOURCES = test-pipe-manager.cpp
test_pipe_manager_CXXFLAGS = $(tests_cxxflags)
test_pipe_manager_LDADD = \
	$(XCORE_LA) $(OCL_LA)  \
	$(NULL)

test_image_blend_SOURCES = test-image-blend.cpp
test_image_blend_CXXFLAGS = $(tests_cxxflags)
test_image_blend_LDADD = \
	$(XCORE_LA) $(OCL_LA)  \
	$(NULL)

test_image_stitching_SOURCES = test-image-stitching.cpp
test_image_stitching_CXXFLAGS = $(tests_cxxflags)
test_image_stitching_LDADD = \
	$(XCORE_LA) $(OCL_LA)  \
	$(NULL)

test_video_stabilization_SOURCES = test-video-stabilization.cpp
test_video_stabilization_CXXFLAGS = $(tests_cxxflags)
test_video_stabilization_LDADD = \
	$(XCORE_LA) $(OCL_LA)  \
	$(NULL)

if HAVE_OPENCV
test_image_deblurring_SOURCES = test-image-deblurring.cpp
test_image_deblurring_CXXFLAGS = $(tests_cxxflags)
	$(NULL)
test_image_deblurring_LDADD = \
	$(XCORE_LA) $(OCL_LA)  \
	$(NULL)

test_image_stitching_CXXFLAGS += $(OPENCV_CFLAGS)
test_image_stitching_LDADD += $(OPENCV_LIBS)
test_video_stabilization_CXXFLAGS += $(OPENCV_CFLAGS)
test_video_stabilization_LDADD += $(OPENCV_LIBS)
test_image_deblurring_CXXFLAGS += $(OPENCV_CFLAGS)
test_image_deblurring_LDADD += $(OPENCV_LIBS)
endif
endif
