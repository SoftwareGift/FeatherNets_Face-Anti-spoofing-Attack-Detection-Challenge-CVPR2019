lib_LTLIBRARIES = libxcam_ocl.la

XCAMOCL_CXXFLAGS = \
    $(XCAM_CXXFLAGS)              \
    $(LIBCL_CFLAGS)               \
    -I$(top_srcdir)/xcore         \
    -I$(top_srcdir)/modules       \
    -I$(top_builddir)/shaders/clx \
    $(NULL)

XCAMOCL_LIBS = \
    $(LIBCL_LIBS) -ldl                    \
    $(top_builddir)/xcore/libxcam_core.la \
    $(NULL)

if HAVE_OPENCV
XCAMOCL_CXXFLAGS += $(OPENCV_CFLAGS)
XCAMOCL_LIBS += \
    $(OPENCV_LIBS)                             \
    $(top_builddir)/modules/ocv/libxcam_ocv.la \
    $(NULL)
endif

if HAVE_LIBDRM
XCAMOCL_CXXFLAGS += $(LIBDRM_CFLAGS)
XCAMOCL_LIBS += \
    -ldrm_intel    \
    $(LIBDRM_LIBS) \
    $(NULL)
endif

xcam_ocl_sources = \
    cl_argument.cpp                   \
    cl_context.cpp                    \
    cl_device.cpp                     \
    cl_kernel.cpp                     \
    cl_memory.cpp                     \
    cl_event.cpp                      \
    cl_utils.cpp                      \
    cl_image_handler.cpp              \
    cl_image_processor.cpp            \
    cl_3a_image_processor.cpp         \
    cl_post_image_processor.cpp       \
    cl_multi_image_handler.cpp        \
    cl_csc_image_processor.cpp        \
    cl_3a_stats_context.cpp           \
    cl_demo_handler.cpp               \
    cl_blender.cpp                    \
    cl_pyramid_blender.cpp            \
    cl_geo_map_handler.cpp            \
    cl_csc_handler.cpp                \
    cl_tnr_handler.cpp                \
    cl_defog_dcp_handler.cpp          \
    cl_bayer_pipe_handler.cpp         \
    cl_bayer_basic_handler.cpp        \
    cl_yuv_pipe_handler.cpp           \
    cl_rgb_pipe_handler.cpp           \
    cl_tonemapping_handler.cpp        \
    cl_newtonemapping_handler.cpp     \
    cl_fisheye_handler.cpp            \
    cl_image_scaler.cpp               \
    cl_image_360_stitch.cpp           \
    cl_retinex_handler.cpp            \
    cl_gauss_handler.cpp              \
    cl_wavelet_denoise_handler.cpp    \
    cl_newwavelet_denoise_handler.cpp \
    cl_wire_frame_handler.cpp         \
    cl_3d_denoise_handler.cpp         \
    cl_image_warp_handler.cpp         \
    cl_video_stabilizer.cpp           \
    cl_video_buffer.cpp               \
    priority_buffer_queue.cpp         \
    $(NULL)

if HAVE_LIBDRM
xcam_ocl_sources += \
    intel/cl_intel_context.cpp \
    intel/cl_va_memory.cpp     \
    cl_image_bo_buffer.cpp     \
    $(NULL)
endif

libxcam_ocl_la_SOURCES = \
    $(xcam_ocl_sources) \
    $(NULL)

libxcam_ocl_la_CXXFLAGS = \
    $(XCAMOCL_CXXFLAGS) \
    $(NULL)

libxcam_ocl_la_LIBADD = \
    $(XCAMOCL_LIBS) \
    $(NULL)

libxcam_ocl_la_LDFLAGS = \
    $(XCAM_LT_LDFLAGS) \
    $(NULL)

libxcam_oclincludedir = $(includedir)/xcam/ocl

nobase_libxcam_oclinclude_HEADERS = \
    cl_argument.h                   \
    cl_context.h                    \
    cl_event.h                      \
    cl_device.h                     \
    cl_memory.h                     \
    cl_kernel.h                     \
    cl_utils.h                      \
    cl_image_handler.h              \
    cl_image_processor.h            \
    priority_buffer_queue.h         \
    cl_3a_image_processor.h         \
    cl_3a_stats_context.h           \
    cl_rgb_pipe_handler.h           \
    cl_bayer_basic_handler.h        \
    cl_bayer_pipe_handler.h         \
    cl_demo_handler.h               \
    cl_tonemapping_handler.h        \
    cl_newtonemapping_handler.h     \
    cl_csc_handler.h                \
    cl_csc_image_processor.h        \
    cl_yuv_pipe_handler.h           \
    cl_tnr_handler.h                \
    cl_post_image_processor.h       \
    cl_multi_image_handler.h        \
    cl_3d_denoise_handler.h         \
    cl_defog_dcp_handler.h          \
    cl_fisheye_handler.h            \
    cl_gauss_handler.h              \
    cl_geo_map_handler.h            \
    cl_image_scaler.h               \
    cl_image_warp_handler.h         \
    cl_image_360_stitch.h           \
    cl_blender.h                    \
    cl_retinex_handler.h            \
    cl_wavelet_denoise_handler.h    \
    cl_newwavelet_denoise_handler.h \
    cl_wire_frame_handler.h         \
    cl_video_stabilizer.h           \
    cl_video_buffer.h               \
    $(NULL)

if HAVE_LIBDRM
nobase_libxcam_oclinclude_HEADERS += \
    intel/cl_intel_context.h \
    intel/cl_va_memory.h     \
    cl_image_bo_buffer.h     \
    $(NULL)
endif

noinst_HEADERS = \
    cl_pyramid_blender.h \
    $(NULL)

libxcam_ocl_la_LIBTOOLFLAGS = --tag=disable-static
