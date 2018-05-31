if HAVE_LIBCL
TESTS_DIR = tests
else
if ENABLE_IA_AIQ
TESTS_DIR = tests
else
TESTS_DIR =
endif
endif

if ENABLE_CAPI
CAPI_DIR = capi
else
CAPI_DIR =
endif

SUBDIRS = xcore shaders modules plugins \
          wrapper $(CAPI_DIR) $(TESTS_DIR) pkgconfig
