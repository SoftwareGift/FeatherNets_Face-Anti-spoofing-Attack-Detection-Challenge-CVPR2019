noinst_PROGRAMS = test-device-manager test-poll-thread

tests_cxxflags = $(XCAM_CXXFLAGS)

if USE_LOCAL_ATOMISP
tests_cxxflags +=               \
    -I$(top_srcdir)/ext/atomisp \
    $(NULL)
endif


test_device_manager_SOURCES = test-device-manager.cpp
test_device_manager_CXXFLAGS = \
	$(tests_cxxflags)          \
	-I$(top_builddir)/xcore    \
	$(NULL)

test_device_manager_LDADD =    \
	$(top_builddir)/xcore/libxcam_core.la \
	$(NULL)

test_poll_thread_SOURCES = test-poll-thread.cpp
test_poll_thread_CXXFLAGS =    \
	$(tests_cxxflags)          \
	-I$(top_builddir)/xcore    \
	$(NULL)

test_poll_thread_LDADD =       \
	$(top_builddir)/xcore/libxcam_core.la \
	$(NULL)
