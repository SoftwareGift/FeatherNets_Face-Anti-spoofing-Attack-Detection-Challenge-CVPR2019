lib_LTLIBRARIES = libxcam_capi.la

XCAMCAPI_CXXFLAGS = \
    $(XCAM_CXXFLAGS)        \
    $(LIBCL_CFLAGS)         \
    -I$(top_srcdir)/xcore   \
    -I$(top_srcdir)/modules \
    $(NULL)

XCAMCAPI_LIBS = \
    $(LIBCL_LIBS) -ldl                         \
    $(top_builddir)/xcore/libxcam_core.la      \
    $(top_builddir)/modules/ocl/libxcam_ocl.la \
    $(NULL)

if HAVE_LIBDRM
XCAMCAPI_CXXFLAGS += $(LIBDRM_CFLAGS)
XCAMCAPI_LIBS += \
    -ldrm_intel    \
    $(LIBDRM_LIBS) \
    $(NULL)
endif

xcam_ocl_sources = \
    xcam_handle.cpp  \
    context_priv.cpp \
    $(NULL)

libxcam_capi_la_SOURCES = \
    $(xcam_ocl_sources) \
    $(NULL)

libxcam_capi_la_CXXFLAGS = \
    $(XCAMCAPI_CXXFLAGS) \
    $(NULL)

libxcam_capi_la_LIBADD = \
    $(XCAMCAPI_LIBS) \
    $(NULL)

libxcam_capi_la_LDFLAGS = \
    $(XCAM_LT_LDFLAGS) \
    $(NULL)

libxcam_capiincludedir = $(includedir)/xcam/capi

nobase_libxcam_capiinclude_HEADERS = \
    xcam_handle.h \
    $(NULL)

libxcam_capi_la_LIBTOOLFLAGS = --tag=disable-static
