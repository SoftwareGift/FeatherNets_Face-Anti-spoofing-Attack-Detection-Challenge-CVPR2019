lib_LTLIBRARIES = libxcam_soft.la

XCAMSOFT_CXXFLAGS =               \
    $(LIBCL_CFLAGS)               \
    -I$(top_srcdir)/xcore         \
    -I$(top_srcdir)/modules       \
    $(NULL)

xcam_soft_sources =                  \
    soft_handler.cpp                 \
    soft_image_allocator.cpp         \
    soft_worker.cpp                  \
   $(NULL)

libxcam_soft_la_SOURCES =   \
    $(xcam_soft_sources)    \
    $(NULL)

libxcam_soft_la_CXXFLAGS =  \
    $(XCAMSOFT_CXXFLAGS)    \
    $(XCAM_CXXFLAGS)        \
    $(NULL)

libxcam_soft_la_LIBADD =                  \
    $(top_builddir)/xcore/libxcam_core.la \
    $(NULL)

libxcam_soft_la_LDFLAGS =   \
    $(XCAM_LT_LDFLAGS)      \
    $(PTHREAD_LDFLAGS)      \
    $(NULL)

libxcam_softincludedir = $(includedir)/xcam/soft

nobase_libxcam_softinclude_HEADERS =   \
    soft_handler.h                     \
    soft_image_allocator.h             \
    soft_worker.h                      \
   $(NULL)

libxcam_soft_la_LIBTOOLFLAGS = --tag=disable-static
