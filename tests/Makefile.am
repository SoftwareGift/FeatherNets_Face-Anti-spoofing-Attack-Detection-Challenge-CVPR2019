noinst_PROGRAMS = \
	test-device-manager  \
	$(NULL)

if ENABLE_IA_AIQ
noinst_PROGRAMS += \
	test-poll-thread     \
	$(NULL)
endif

if HAVE_LIBCL
noinst_PROGRAMS +=      \
	test-cl-image       \
	test-binary-kernel  \
	test-pipe-manager   \
	test-image-blend    \
	$(NULL)
endif

XCORE_DIR = $(top_builddir)/xcore
XCORE_LA = $(top_builddir)/xcore/libxcam_core.la

ISP_DIR = $(top_builddir)/modules/isp
ISP_LA = $(top_builddir)/modules/isp/libxcam_isp.la

OCL_DIR = $(top_builddir)/modules/ocl
OCL_LA = $(top_builddir)/modules/ocl/libxcam_ocl.la

tests_cxxflags = $(XCAM_CXXFLAGS)
if HAVE_LIBDRM
tests_cxxflags += $(LIBDRM_CFLAGS) $(LIBDRM_LIBS)
endif

if USE_LOCAL_ATOMISP
tests_cxxflags += \
	-I$(top_srcdir)/ext/atomisp  \
	$(NULL)
endif

test_device_manager_SOURCES = test-device-manager.cpp
test_device_manager_CXXFLAGS = $(tests_cxxflags) -I$(XCORE_DIR)
test_device_manager_LDADD = $(XCORE_LA)

if ENABLE_IA_AIQ
test_device_manager_CXXFLAGS += -I$(ISP_DIR)
test_device_manager_LDADD += $(ISP_LA)

test_poll_thread_SOURCES = test-poll-thread.cpp
test_poll_thread_CXXFLAGS = \
	$(tests_cxxflags) -I$(XCORE_DIR) -I$(ISP_DIR)  \
	$(NULL)
test_poll_thread_LDADD = \
	$(XCORE_LA) $(ISP_LA)  \
	$(NULL)
endif

if HAVE_LIBCL
test_device_manager_CXXFLAGS += -I$(OCL_DIR)
test_device_manager_LDADD += $(OCL_LA)

test_cl_image_SOURCES = test-cl-image.cpp
test_cl_image_CXXFLAGS = \
	$(tests_cxxflags) -I$(XCORE_DIR) -I$(OCL_DIR)  \
	$(NULL)
test_cl_image_LDADD = \
	$(XCORE_LA) $(OCL_LA)  \
	$(NULL)

test_binary_kernel_SOURCES = test-binary-kernel.cpp
test_binary_kernel_CXXFLAGS = \
	$(tests_cxxflags) -I$(XCORE_DIR) -I$(OCL_DIR)  \
	$(NULL)
test_binary_kernel_LDADD = \
	$(XCORE_LA) $(OCL_LA)  \
	$(NULL)

test_pipe_manager_SOURCES = test-pipe-manager.cpp
test_pipe_manager_CXXFLAGS = \
	$(tests_cxxflags) -I$(XCORE_DIR) -I$(OCL_DIR)  \
	$(NULL)
test_pipe_manager_LDADD = \
	$(XCORE_LA) $(OCL_LA)  \
	$(NULL)

test_image_blend_SOURCES = test-image-blend.cpp
test_image_blend_CXXFLAGS = \
	$(tests_cxxflags) -I$(XCORE_DIR) -I$(OCL_DIR)  \
	$(NULL)
test_image_blend_LDADD = \
	$(XCORE_LA) $(OCL_LA)  \
	$(NULL)

endif
