lib_LTLIBRARIES = libxcam_core.la

PTHREAD_LDFLAGS = -pthread

XCAM_CORE_CXXFLAGS = $(XCAM_CXXFLAGS)
XCAM_CORE_LIBS = -ldl  \
    $(NULL)

xcam_sources = \
    analyzer_loader.cpp                 \
    smart_analyzer_loader.cpp           \
    buffer_pool.cpp                     \
    device_manager.cpp                  \
    pipe_manager.cpp                    \
    dynamic_analyzer.cpp                \
    dynamic_analyzer_loader.cpp         \
    smart_analyzer.cpp                  \
    smart_analysis_handler.cpp          \
    fake_poll_thread.cpp                \
    handler_interface.cpp               \
    image_processor.cpp                 \
    image_file_handle.cpp               \
    poll_thread.cpp                     \
    swapped_buffer.cpp                  \
    uvc_device.cpp                      \
    v4l2_buffer_proxy.cpp               \
    v4l2_device.cpp                     \
    video_buffer.cpp                    \
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
    x3a_analyze_tuner.cpp               \
    x3a_ciq_tuning_handler.cpp          \
    x3a_ciq_tnr_tuning_handler.cpp      \
    x3a_ciq_bnr_ee_tuning_handler.cpp   \
    x3a_ciq_wavelet_tuning_handler.cpp  \
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
    $(PTHREAD_LDFLAGS)  \
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
    device_manager.h               \
    pipe_manager.h                 \
    handler_interface.h            \
    image_processor.h              \
    image_file_handle.h            \
    safe_list.h                    \
    smartptr.h                     \
    swapped_buffer.h               \
    v4l2_buffer_proxy.h            \
    v4l2_device.h                  \
    video_buffer.h                 \
    xcam_analyzer.h                \
    x3a_analyzer.h                 \
    x3a_analyzer_manager.h         \
    x3a_event.h                    \
    x3a_image_process_center.h     \
    x3a_result.h                   \
    xcam_mutex.h                   \
    xcam_thread.h                  \
    xcam_utils.h                   \
    $(NULL)

if HAVE_LIBDRM
nobase_libxcam_coreinclude_HEADERS += \
    drm_display.h      \
    drm_v4l2_buffer.h  \
    $(NULL)
endif
