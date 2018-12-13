SUMMARY = "Libxcam"
DESCRIPTION = "Libxcam: Extended camera features and cross platform computer vision project"
HOMEPAGE = "https://github.com/intel/libxcam/wiki"
LICENSE = "Apache-2.0"

PR = "r0"
S = "${WORKDIR}/git"

LIC_FILES_CHKSUM = "file://${S}/LICENSE;md5=a739187a9544e0731270d11a8f5be792"

SRC_URI = "git://github.com/intel/libxcam.git;branch=master"
SRCREV = "${AUTOREV}"

DEPENDS = "glib-2.0 libdrm beignet opencv gstreamer1.0 gstreamer1.0-plugins-base"

inherit autotools pkgconfig

EXTRA_OECONF = "--enable-gst --enable-drm --enable-libcl --enable-smartlib --enable-opencv"

CFLAGS += "-fPIE -fPIC"
CFLAGS += "-O2 -D_FORTIFY_SOURCE=2"
CFLAGS += "-Wall -Wno-unused-parameter"
CFLAGS += "-fstack-protector"

LDFLAGS += "-z noexecstack"
LDFLAGS += "-z relro -z now"

PACKAGES += "${PN}-test"

FILES_${PN} += "${libdir}/libxcam_core.so.*"
FILES_${PN} += "${libdir}/libxcam_ocl.so.*"
FILES_${PN} += "${libdir}/gstreamer-1.0/libgstxcamfilter.*"

FILES_${PN}-dev += "${includedir}/xcam/*"
FILES_${PN}-dev += "${libdir}/pkgconfig/libxcam.pc"
FILES_${PN}-dev += "${libdir}/libxcam_core.so"
FILES_${PN}-dev += "${libdir}/libxcam_core.la"
FILES_${PN}-dev += "${libdir}/libxcam_core.a"
FILES_${PN}-dev += "${libdir}/libxcam_ocl.so"
FILES_${PN}-dev += "${libdir}/libxcam_ocl.la"

FILES_${PN}-test = "${bindir}/test-*"

FILES_${PN}-dbg += "${libdir}/gstreamer-1.0/.debug/*"
FILES_${PN}-dbg += "${libdir}/.debug/"
FILES_${PN}-dbg += "${bindir}/.debug/"

