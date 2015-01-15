pcverfiles = xcam_core.pc

all-local: $(pcverfiles)

pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = $(pcverfiles)

CLEANFILES = $(pcverfiles)
pcinfiles =  xcam_core.pc.in

DISTCLEANFILES = $(pcinfiles:.in=)
EXTRA_DIST = $(pcinfiles)
