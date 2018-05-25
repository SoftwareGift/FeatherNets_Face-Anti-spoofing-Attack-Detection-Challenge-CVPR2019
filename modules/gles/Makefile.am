lib_LTLIBRARIES = libxcam_gles.la

XCAM_GLES_CXXFLAGS =                 \
    $(LIBGL_CFLAGS)                  \
    -I$(top_srcdir)/xcore            \
    -I$(top_srcdir)/modules          \
    -I$(top_builddir)/glx_shader     \
    $(NULL)

XCAM_GLES_LIBS =                     \
    $(LIBGL_LIBS)                    \
    $(NULL)

xcam_gles_sources =                  \
    gles_std.cpp                     \
    gl_buffer.cpp                    \
    gl_program.cpp                   \
    gl_shader.cpp                    \
    gl_video_buffer.cpp              \
    gl_compute_program.cpp           \
    gl_command.cpp                   \
    gl_utils.cpp                     \
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
    gl_video_buffer.h                \
    gl_compute_program.h             \
    gl_command.h                     \
    gl_utils.h                       \
    $(NULL)

noinst_HEADERS =                     \
    $(NULL)

libxcam_gles_la_LIBTOOLFLAGS = --tag=disable-static
