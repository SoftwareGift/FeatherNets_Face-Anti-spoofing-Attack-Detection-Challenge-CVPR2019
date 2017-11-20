LOCAL_PATH:= $(call my-dir)

XCAM_CFLAGS := -fPIC -W -Wall -D_REENTRANT -Wformat -Wno-unused-parameter -Wformat-security -fstack-protector
ifeq ($(ENABLE_DEBUG), 1)
XCAM_CFLAGS += -DDEBUG
endif


# For libxcam
# =================================================

include $(CLEAR_VARS)

LOCAL_MODULE := libxcam
LOCAL_MODULE_TAGS := optional

XCAM_XCORE_SRC_FILES := \
    xcore/buffer_pool.cpp \
    xcore/calibration_parser.cpp \
    xcore/file_handle.cpp \
    xcore/image_file_handle.cpp \
    xcore/image_handler.cpp \
    xcore/surview_fisheye_dewarp.cpp \
    xcore/thread_pool.cpp \
    xcore/video_buffer.cpp \
    xcore/worker.cpp \
    xcore/xcam_buffer.cpp \
    xcore/xcam_common.cpp \
    xcore/xcam_thread.cpp \
    xcore/xcam_utils.cpp \
    xcore/interface/blender.cpp \
    xcore/interface/feature_match.cpp \
    xcore/interface/geo_mapper.cpp \
    xcore/interface/stitcher.cpp \
    $(NULL)

XCAM_SOFT_SRC_FILES := \
    modules/soft/soft_blender.cpp \
    modules/soft/soft_blender_tasks_priv.cpp \
    modules/soft/soft_copy_task.cpp \
    modules/soft/soft_geo_mapper.cpp \
    modules/soft/soft_handler.cpp \
    modules/soft/soft_stitcher.cpp \
    modules/soft/soft_video_buf_allocator.cpp \
    modules/soft/soft_worker.cpp \
    $(NULL)

LOCAL_SRC_FILES := $(XCAM_XCORE_SRC_FILES) $(XCAM_SOFT_SRC_FILES)

LOCAL_C_INCLUDES := \
    $(LOCAL_PATH)/xcore \
    $(LOCAL_PATH)/modules \
    $(NULL)

LOCAL_CFLAGS := $(XCAM_CFLAGS)
LOCAL_CPPFLAGS := $(LOCAL_CFLAGS) -frtti

include $(BUILD_SHARED_LIBRARY)


# For test-soft-image
# =================================================

include $(CLEAR_VARS)

LOCAL_MODULE := test-soft-image
LOCAL_MODULE_TAGS := optional

LOCAL_SHARED_LIBRARIES := libxcam

LOCAL_SRC_FILES := \
    tests/test-soft-image.cpp
    $(NULL)

LOCAL_C_INCLUDES := \
    $(LOCAL_PATH)/xcore \
    $(LOCAL_PATH)/modules \
    $(LOCAL_PATH)/tests \
    $(NULL)

LOCAL_CFLAGS := $(XCAM_CFLAGS)
LOCAL_CPPFLAGS := $(LOCAL_CFLAGS)

include $(BUILD_EXECUTABLE)

