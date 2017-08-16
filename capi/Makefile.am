lib_LTLIBRARIES = libxcam_capi.la

XCAMCAPI_LIBS =                    \
    $(LIBCL_LIBS)                  \
    -ldl                           \
    $(NULL)

XCAMCAPI_CXXFLAGS =                \
    $(XCAM_CXXFLAGS)               \
    $(LIBCL_CFLAGS)                \
    -I$(top_srcdir)/xcore          \
    -I$(top_srcdir)/modules        \
    $(NULL)

if HAVE_LIBDRM
XCAMCAPI_CXXFLAGS += $(LIBDRM_CFLAGS)
XCAMCAPI_LIBS +=                   \
    -ldrm_intel                    \
    $(LIBDRM_LIBS)                 \
    $(NULL)
endif

xcam_ocl_sources =                 \
    xcam_handle.cpp                \
    context_priv.cpp               \
   $(NULL)

libxcam_capi_la_SOURCES =          \
    $(xcam_ocl_sources)            \
    $(NULL)

libxcam_capi_la_CXXFLAGS =       \
    $(XCAMCAPI_CXXFLAGS)         \
    $(NULL)

libxcam_capi_la_LIBADD =           \
    $(top_builddir)/modules/ocl/libxcam_ocl.la \
    $(top_builddir)/xcore/libxcam_core.la      \
    $(XCAMCAPI_LIBS)                           \
    $(NULL)

libxcam_capi_la_LDFLAGS =          \
    $(XCAM_LT_LDFLAGS)             \
    $(PTHREAD_LDFLAGS)             \
    $(NULL)

libxcam_capiincludedir = $(includedir)/xcam/capi

nobase_libxcam_capiinclude_HEADERS = \
    xcam_handle.h                    \
   $(NULL)

libxcam_capi_la_LIBTOOLFLAGS = --tag=disable-static
