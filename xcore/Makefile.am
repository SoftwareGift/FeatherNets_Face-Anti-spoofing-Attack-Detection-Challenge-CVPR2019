lib_LTLIBRARIES = libxcam_core.la

XCAM_CORE_CXXFLAGS = \
    $(XCAM_CXXFLAGS) \
    -pthread         \
    $(NULL)
XCAM_CORE_LIBS = \
    -ldl \
    $(NULL)

xcam_sources = \
    analyzer_loader.cpp                 \
    smart_analyzer_loader.cpp           \
    buffer_pool.cpp                     \
    calibration_parser.cpp              \
    device_manager.cpp                  \
    pipe_manager.cpp                    \
    dma_video_buffer.cpp                \
    dynamic_analyzer.cpp                \
    dynamic_analyzer_loader.cpp         \
    smart_analyzer.cpp                  \
    smart_analysis_handler.cpp          \
    smart_buffer_priv.cpp               \
    fake_poll_thread.cpp                \
    file_handle.cpp                     \
    handler_interface.cpp               \
    image_handler.cpp                   \
    image_processor.cpp                 \
    image_projector.cpp                 \
    image_file_handle.cpp               \
    poll_thread.cpp                     \
    surview_fisheye_dewarp.cpp          \
    swapped_buffer.cpp                  \
    thread_pool.cpp                     \
    uvc_device.cpp                      \
    v4l2_buffer_proxy.cpp               \
    v4l2_device.cpp                     \
    video_buffer.cpp                    \
    once_map_video_buffer_priv.cpp      \
    worker.cpp                          \
    xcam_analyzer.cpp                   \
    x3a_analyzer.cpp                    \
    x3a_analyzer_manager.cpp            \
    x3a_analyzer_simple.cpp             \
    x3a_image_process_center.cpp        \
    x3a_stats_pool.cpp                  \
    x3a_result.cpp                      \
    x3a_result_factory.cpp              \
    xcam_common.cpp                     \
    xcam_buffer.cpp                     \
    xcam_thread.cpp                     \
    xcam_utils.cpp                      \
    interface/feature_match.cpp         \
    interface/blender.cpp               \
    interface/geo_mapper.cpp            \
    interface/stitcher.cpp              \
    $(NULL)

if HAVE_LIBDRM
XCAM_CORE_CXXFLAGS += $(LIBDRM_CFLAGS)
XCAM_CORE_LIBS += \
    -ldrm_intel          \
    $(LIBDRM_LIBS)       \
    $(NULL)

xcam_sources += \
    drm_bo_buffer.cpp    \
    drm_display.cpp      \
    drm_v4l2_buffer.cpp  \
    $(NULL)
endif

libxcam_core_la_CXXFLAGS = \
    $(XCAM_CORE_CXXFLAGS)  \
    $(NULL)

libxcam_core_la_SOURCES = \
    $(xcam_sources)  \
    $(NULL)

libxcam_core_la_LDFLAGS = \
    -no-undefined       \
    -pthread            \
    $(XCAM_LT_LDFLAGS)  \
    $(NULL)

libxcam_core_la_LIBADD = \
    $(XCAM_CORE_LIBS)  \
    $(NULL)

libxcam_coreincludedir = $(includedir)/xcam

nobase_libxcam_coreinclude_HEADERS = \
    base/xcam_3a_result.h          \
    base/xcam_3a_types.h           \
    base/xcam_3a_description.h     \
    base/xcam_buffer.h             \
    base/xcam_params.h             \
    base/xcam_common.h             \
    base/xcam_defs.h               \
    base/xcam_smart_description.h  \
    base/xcam_smart_result.h       \
    calibration_parser.h           \
    device_manager.h               \
    dma_video_buffer.h             \
    file_handle.h                  \
    pipe_manager.h                 \
    handler_interface.h            \
    image_handler.h                \
    image_processor.h              \
    image_projector.h              \
    image_file_handle.h            \
    safe_list.h                    \
    smartptr.h                     \
    surview_fisheye_dewarp.h       \
    swapped_buffer.h               \
    thread_pool.h                  \
    v4l2_buffer_proxy.h            \
    v4l2_device.h                  \
    video_buffer.h                 \
    worker.h                       \
    xcam_analyzer.h                \
    x3a_analyzer.h                 \
    x3a_analyzer_manager.h         \
    x3a_event.h                    \
    x3a_image_process_center.h     \
    x3a_result.h                   \
    xcam_mutex.h                   \
    xcam_thread.h                  \
    xcam_std.h                     \
    xcam_utils.h                   \
    xcam_obj_debug.h               \
    buffer_pool.h                  \
    meta_data.h                    \
    vec_mat.h                      \
    interface/data_types.h         \
    interface/feature_match.h      \
    interface/blender.h            \
    interface/geo_mapper.h         \
    interface/stitcher.h           \
    $(NULL)

if HAVE_LIBDRM
nobase_libxcam_coreinclude_HEADERS += \
    drm_bo_buffer.h    \
    drm_display.h      \
    drm_v4l2_buffer.h  \
    $(NULL)
endif
