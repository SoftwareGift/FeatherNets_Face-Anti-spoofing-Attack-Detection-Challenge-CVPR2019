if ENABLE_IA_AIQ
AIQ_DIR = aiq
else
AIQ_DIR =
endif

if ENABLE_IA_AIQ
if HAVE_LIBCL
HYBRID_DIR = hybrid
else
HYBRID_DIR =
endif
endif

SUBDIRS = $(AIQ_DIR) $(HYBRID_DIR)
