if HAVE_OPENCV
OCV_DIR = ocv
else
OCV_DIR =
endif

if HAVE_LIBCL
OCL_DIR = ocl
else
OCL_DIR =
endif

if ENABLE_IA_AIQ
ISP_DIR = isp
else
ISP_DIR =
endif

if HAVE_GLES
GLES_DIR = gles
else
GLES_DIR =
endif

if HAVE_VULKAN
VULKAN_DIR = vulkan
else
VULKAN_DIR =
endif

if ENABLE_RENDER
RENDER_DIR = render
else
RENDER_DIR =
endif

SUBDIRS = $(OCV_DIR) soft $(ISP_DIR) $(OCL_DIR) $(GLES_DIR) $(VULKAN_DIR) $(RENDER_DIR)