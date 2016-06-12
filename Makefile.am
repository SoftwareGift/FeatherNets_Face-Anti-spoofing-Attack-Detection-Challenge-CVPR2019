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

SUBDIRS = $(CLX_KERNEL_DIR) $(CLX_BIN_DIR) modules xcore plugins \
          wrapper tests pkgconfig
