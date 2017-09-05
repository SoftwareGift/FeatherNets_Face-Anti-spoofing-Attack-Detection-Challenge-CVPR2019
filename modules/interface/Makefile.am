lib_LTLIBRARIES = libxcam_interface.la

XCAM_INTERFACE_CXXFLAGS =              \
    $(LIBCL_CFLAGS)                    \
    -I$(top_srcdir)/xcore              \
    -I$(top_srcdir)/modules            \
    $(NULL)

xcam_interface_sources =               \
    blender.cpp                        \
   $(NULL)

libxcam_interface_la_SOURCES =         \
    $(xcam_interface_sources)          \
    $(NULL)

libxcam_interface_la_CXXFLAGS =        \
    $(XCAM_INTERFACE_CXXFLAGS)         \
    $(XCAM_CXXFLAGS)                   \
    $(NULL)

libxcam_interface_la_LIBADD =             \
    $(top_builddir)/xcore/libxcam_core.la \
    $(NULL)

libxcam_interface_la_LDFLAGS =   \
    $(XCAM_LT_LDFLAGS)           \
    $(PTHREAD_LDFLAGS)           \
    $(NULL)

libxcam_interfaceincludedir = $(includedir)/xcam/interface

nobase_libxcam_interfaceinclude_HEADERS =  \
    data_types.h                       \
    blender.h                          \
    $(NULL)

libxcam_interface_la_LIBTOOLFLAGS = --tag=disable-static
