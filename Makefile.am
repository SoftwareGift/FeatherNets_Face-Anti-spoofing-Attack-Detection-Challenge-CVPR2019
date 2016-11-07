if HAVE_LIBCL
CLX_KERNEL_DIR = clx_kernel
CLX_BIN_DIR = clx_bin
else
CLX_KERNEL_DIR =
CLX_BIN_DIR =
endif

if HAVE_LIBCL
TESTS_DIR = tests
else
if ENABLE_IA_AIQ
TESTS_DIR = tests
else
TESTS_DIR =
endif
endif

SUBDIRS = xcore $(CLX_KERNEL_DIR) $(CLX_BIN_DIR) modules plugins \
          wrapper $(TESTS_DIR) pkgconfig
