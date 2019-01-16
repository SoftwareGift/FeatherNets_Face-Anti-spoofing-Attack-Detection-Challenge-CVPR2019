lib_LTLIBRARIES = libxcam_vulkan.la

XCAM_VK_CXXFLAGS = \
    $(XCAM_CXXFLAGS)              \
    $(LIBVULKAN_CFLAGS)           \
    -I$(top_srcdir)/xcore         \
    -I$(top_srcdir)/modules       \
    -I$(top_builddir)/shaders/spv \
    $(NULL)

XCAM_VK_LIBS = \
    $(LIBVULKAN_LIBS)                     \
    $(top_builddir)/xcore/libxcam_core.la \
    $(NULL)

if HAVE_OPENCV
XCAM_VK_LIBS += $(top_builddir)/modules/ocv/libxcam_ocv.la
endif

xcam_vulkan_sources = \
    vk_cmdbuf.cpp              \
    vk_descriptor.cpp          \
    vk_device.cpp              \
    vk_handler.cpp             \
    vk_instance.cpp            \
    vk_memory.cpp              \
    vk_pipeline.cpp            \
    vk_shader.cpp              \
    vk_sync.cpp                \
    vk_video_buf_allocator.cpp \
    vk_worker.cpp              \
    vulkan_common.cpp          \
    vk_copy_handler.cpp        \
    vk_geomap_handler.cpp      \
    vk_blender.cpp             \
    vk_stitcher.cpp            \
    $(NULL)

libxcam_vulkan_la_SOURCES = \
    $(xcam_vulkan_sources) \
    $(NULL)

libxcam_vulkan_la_CXXFLAGS = \
    $(XCAM_VK_CXXFLAGS) \
    $(NULL)

libxcam_vulkan_la_LIBADD = \
    $(XCAM_VK_LIBS) \
    $(NULL)

libxcam_vulkan_la_LDFLAGS = \
    $(XCAM_LT_LDFLAGS) \
    $(NULL)

libxcam_vulkanincludedir = $(includedir)/xcam/vulkan

nobase_libxcam_vulkaninclude_HEADERS = \
    vk_cmdbuf.h              \
    vk_descriptor.h          \
    vk_device.h              \
    vk_handler.h             \
    vk_instance.h            \
    vk_memory.h              \
    vk_pipeline.h            \
    vk_shader.h              \
    vk_sync.h                \
    vk_video_buf_allocator.h \
    vk_worker.h              \
    vulkan_common.h          \
    vk_copy_handler.h        \
    vk_geomap_handler.h      \
    vk_blender.h             \
    vk_stitcher.h            \
    $(NULL)

libxcam_vulkan_la_LIBTOOLFLAGS = --tag=disable-static
