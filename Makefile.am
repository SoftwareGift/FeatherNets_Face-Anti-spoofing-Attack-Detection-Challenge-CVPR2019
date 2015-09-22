
if ENABLE_3ALIB
X3ALIB_DIR = 3alib
else
X3ALIB_DIR =
endif

if HAVE_LIBCL
CLX_KERNEL_DIR = clx_kernel
else
CLX_KERNEL_DIR =
endif

if HAVE_LIBCL
CLX_BIN_DIR = clx_bin
else
CLX_BIN_DIR =
endif

if ENABLE_SMART_LIB
SMART_ANALYSIS_DIR = smartlib
else
SMART_ANALYSIS_DIR =
endif

SUBDIRS = $(CLX_KERNEL_DIR) $(CLX_BIN_DIR) xcore wrapper tests \
          $(X3ALIB_DIR) $(SMART_ANALYSIS_DIR) pkgconfig
