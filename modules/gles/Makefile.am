lib_LTLIBRARIES = libxcam_gles.la

XCAM_GLES_CXXFLAGS =                 \
    $(LIBGL_CFLAGS)                  \
    -I$(top_srcdir)/xcore            \
    -I$(top_srcdir)/modules          \
    $(NULL)

XCAM_GLES_LIBS =                     \
    $(LIBGL_LIBS)                    \
    $(NULL)

xcam_gles_sources =                  \
    gl_buffer.cpp                    \
    gl_program.cpp                   \
    gl_shader.cpp                    \
    gl_video_buffer_pool.cpp         \
    $(NULL)

libxcam_gles_la_SOURCES =    \
    $(xcam_gles_sources)     \
    $(NULL)

libxcam_gles_la_CXXFLAGS =   \
    $(XCAM_GLES_CXXFLAGS)    \
    $(XCAM_CXXFLAGS)         \
    $(NULL)

libxcam_gles_la_LIBADD =                  \
    $(top_builddir)/xcore/libxcam_core.la \
    $(XCAM_GLES_LIBS)                     \
    $(NULL)

libxcam_gles_la_LDFLAGS =   \
    $(XCAM_LT_LDFLAGS)      \
    $(PTHREAD_LDFLAGS)      \
    $(NULL)

libxcam_glesincludedir = $(includedir)/xcam/gles

nobase_libxcam_glesinclude_HEADERS = \
    gles_std.h                       \
    gl_buffer.h                      \
    gl_program.h                     \
    gl_shader.h                      \
    gl_video_buffer_pool.h           \
    $(NULL)

noinst_HEADERS =                     \
    $(NULL)

libxcam_gles_la_LIBTOOLFLAGS = --tag=disable-static
