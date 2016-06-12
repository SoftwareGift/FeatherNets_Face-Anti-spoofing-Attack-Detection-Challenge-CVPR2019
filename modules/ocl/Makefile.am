module_LTLIBRARIES = libxcam_ocl.la

PTHREAD_LDFLAGS = -pthread

XCAMOCL_CXXFLAGS = $(XCAM_CXXFLAGS)
XCAMOCL_LIBS = -ldl  \
    $(NULL)

XCAMOCL_CXXFLAGS += \
    $(LIBCL_CFLAGS)             \
    -I$(top_srcdir)/clx_kernel  \
    -I$(top_srcdir)/clx_bin     \
    $(NULL)

XCAMOCL_LIBS += \
    $(LIBCL_LIBS)  \
    $(NULL)

if HAVE_LIBDRM
XCAMOCL_CXXFLAGS += $(LIBDRM_CFLAGS)
XCAMOCL_LIBS += \
    -ldrm_intel     \
    $(LIBDRM_LIBS)  \
    $(NULL)
endif

xcam_ocl_sources = \
    cl_context.cpp                     \
    cl_device.cpp                      \
    cl_kernel.cpp                      \
    cl_memory.cpp                      \
    cl_event.cpp                       \
    cl_image_bo_buffer.cpp             \
    cl_image_handler.cpp               \
    cl_image_processor.cpp             \
    cl_3a_image_processor.cpp          \
    cl_post_image_processor.cpp        \
    cl_csc_image_processor.cpp         \
    cl_3a_stats_calculator.cpp         \
    cl_3a_stats_context.cpp            \
    cl_demo_handler.cpp                \
    cl_blc_handler.cpp                 \
    cl_hdr_handler.cpp                 \
    cl_demosaic_handler.cpp            \
    cl_csc_handler.cpp                 \
    cl_wb_handler.cpp                  \
    cl_denoise_handler.cpp             \
    cl_gamma_handler.cpp               \
    cl_snr_handler.cpp                 \
    cl_macc_handler.cpp                \
    cl_tnr_handler.cpp                 \
    cl_ee_handler.cpp                  \
    cl_dpc_handler.cpp                 \
    cl_defog_dcp_handler.cpp           \
    cl_bnr_handler.cpp                 \
    cl_bayer_pipe_handler.cpp          \
    cl_bayer_basic_handler.cpp         \
    cl_yuv_pipe_handler.cpp            \
    cl_rgb_pipe_handler.cpp            \
    cl_tonemapping_handler.cpp         \
    cl_newtonemapping_handler.cpp      \
    cl_biyuv_handler.cpp               \
    cl_image_scaler.cpp                \
    cl_retinex_handler.cpp             \
    cl_gauss_handler.cpp               \
    cl_wavelet_denoise_handler.cpp     \
    cl_newwavelet_denoise_handler.cpp  \
    cl_wire_frame_handler.cpp          \
    priority_buffer_queue.cpp          \
    $(NULL)

moduledir="$(libdir)/xcam"

libxcam_ocl_la_SOURCES = \
    $(xcam_ocl_sources)    \
    $(NULL)

libxcam_ocl_la_CXXFLAGS = \
    $(XCAMOCL_CXXFLAGS)            \
    -I$(top_builddir)/xcore        \
    -I$(top_builddir)/modules/isp  \
    -I$(top_builddir)/modules/ocl  \
    $(NULL)

libxcam_ocl_la_LIBADD = \
    $(XCAMOCL_LIBS)                \
    $(NULL)

libxcam_ocl_la_LDFLAGS = \
    -avoid-version                 \
    $(PTHREAD_LDFLAGS)             \
    $(NULL)

libxcam_ocl_la_LIBTOOLFLAGS = --tag=disable-static
