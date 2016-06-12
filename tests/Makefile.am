noinst_PROGRAMS = test-device-manager test-poll-thread

if HAVE_LIBCL
noinst_PROGRAMS += test-cl-image test-binary-kernel
endif

tests_cxxflags = $(XCAM_CXXFLAGS)

if HAVE_LIBDRM
tests_cxxflags += $(LIBDRM_CFLAGS) $(LIBDRM_LIBS)
endif


if USE_LOCAL_ATOMISP
tests_cxxflags +=               \
    -I$(top_srcdir)/ext/atomisp \
    $(NULL)
endif


test_device_manager_SOURCES = test-device-manager.cpp
test_device_manager_CXXFLAGS = \
	$(tests_cxxflags)              \
	-I$(top_builddir)/xcore        \
	-I$(top_builddir)/modules/isp  \
	-I$(top_builddir)/modules/ocl  \
	$(NULL)

test_device_manager_LDADD =    \
	$(top_builddir)/xcore/libxcam_core.la       \
	$(top_builddir)/modules/isp/libxcam_isp.la  \
	$(top_builddir)/modules/ocl/libxcam_ocl.la  \
	$(NULL)

test_poll_thread_SOURCES = test-poll-thread.cpp
test_poll_thread_CXXFLAGS = \
	$(tests_cxxflags)              \
	-I$(top_builddir)/xcore        \
	-I$(top_builddir)/modules/isp  \
	$(NULL)

test_poll_thread_LDADD = \
	$(top_builddir)/xcore/libxcam_core.la       \
	$(top_builddir)/modules/isp/libxcam_isp.la  \
	$(NULL)

if HAVE_LIBCL
test_cl_image_SOURCES = test-cl-image.cpp
test_cl_image_CXXFLAGS = \
	$(tests_cxxflags)              \
	-I$(top_builddir)/xcore        \
	-I$(top_builddir)/modules/isp  \
	-I$(top_builddir)/modules/ocl  \
	$(NULL)

test_cl_image_LDADD = \
	$(top_builddir)/xcore/libxcam_core.la       \
	$(top_builddir)/modules/isp/libxcam_isp.la  \
	$(top_builddir)/modules/ocl/libxcam_ocl.la  \
	$(NULL)

test_binary_kernel_SOURCES = test-binary-kernel.cpp
test_binary_kernel_CXXFLAGS = \
	$(tests_cxxflags)              \
	-I$(top_builddir)/xcore        \
	-I$(top_builddir)/modules/ocl  \
	$(NULL)

test_binary_kernel_LDADD = \
	$(top_builddir)/xcore/libxcam_core.la       \
	$(top_builddir)/modules/ocl/libxcam_ocl.la  \
	$(NULL)
endif
