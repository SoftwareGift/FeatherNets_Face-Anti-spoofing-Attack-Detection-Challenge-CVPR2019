noinst_PROGRAMS = \
    $(NULL)

if ENABLE_IA_AIQ
noinst_PROGRAMS += \
    test-device-manager  \
    test-poll-thread     \
    $(NULL)
endif

if HAVE_LIBCL
noinst_PROGRAMS += \
    test-cl-image       \
    test-binary-kernel  \
    test-pipe-manager   \
    $(NULL)
endif

XCORE_DIR = $(top_builddir)/xcore
XCORE_LA = $(top_builddir)/xcore/libxcam_core.la

ISP_DIR = $(top_builddir)/modules/isp
ISP_LA = $(top_builddir)/modules/isp/libxcam_isp.la

if HAVE_LIBCL
OCL_DIR = $(top_builddir)/modules/ocl
OCL_LA = $(top_builddir)/modules/ocl/libxcam_ocl.la
else
OCL_DIR =
OCL_LA =
endif

tests_cxxflags = $(XCAM_CXXFLAGS)

if HAVE_LIBDRM
tests_cxxflags += $(LIBDRM_CFLAGS) $(LIBDRM_LIBS)
endif

if USE_LOCAL_ATOMISP
tests_cxxflags += \
    -I$(top_srcdir)/ext/atomisp  \
    $(NULL)
endif

if ENABLE_IA_AIQ
test_device_manager_SOURCES = test-device-manager.cpp
test_device_manager_CXXFLAGS = \
	$(tests_cxxflags) -I$(XCORE_DIR)  \
	-I$(ISP_DIR) -I$(OCL_DIR)         \
	$(NULL)
test_device_manager_LDADD = \
	$(XCORE_LA) $(ISP_LA) $(OCL_LA)  \
	$(NULL)

test_poll_thread_SOURCES = test-poll-thread.cpp
test_poll_thread_CXXFLAGS = \
	$(tests_cxxflags) -I$(XCORE_DIR) -I$(ISP_DIR)  \
	$(NULL)
test_poll_thread_LDADD = \
	$(XCORE_LA) $(ISP_LA)  \
	$(NULL)
endif

if HAVE_LIBCL
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
endif
