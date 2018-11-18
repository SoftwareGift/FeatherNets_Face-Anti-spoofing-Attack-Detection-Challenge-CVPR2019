lib_LTLIBRARIES = libxcam_render.la

XCAM_RENDER_CXXFLAGS =               \
    $(LIBOSG_CFLAGS)                  \
    -I$(top_srcdir)/xcore            \
    -I$(top_srcdir)/modules          \
    $(NULL)

XCAM_RENDER_LIBS =                   \
    $(losgGA)                        \
    -losgDB                          \
    -losgUtil                        \
    -losgFX                          \
    -losgText                        \
    -losgViewer                      \
    -losg                            \
    $(NULL)

xcam_render_sources =                 \
    render_osg_camera_manipulator.cpp \
    render_osg_model.cpp              \
    render_osg_viewer.cpp             \
    $(NULL)

libxcam_render_la_SOURCES =    \
    $(xcam_render_sources)     \
    $(NULL)

libxcam_render_la_CXXFLAGS =   \
    $(XCAM_RENDER_CXXFLAGS)    \
    $(XCAM_CXXFLAGS)           \
    $(NULL)

libxcam_render_la_LIBADD =                  \
    $(top_builddir)/xcore/libxcam_core.la   \
    $(XCAM_RENDER_LIBS)                     \
    $(NULL)

libxcam_render_la_LDFLAGS =   \
    $(XCAM_LT_LDFLAGS)        \
    $(PTHREAD_LDFLAGS)        \
    $(NULL)

libxcam_renderincludedir = $(includedir)/xcam/render

nobase_libxcam_renderinclude_HEADERS =  \
    render_osg_camera_manipulator.h     \
    render_osg_model.h                  \
    render_osg_viewer.h                 \
    render_osg_shader.h                 \
    $(NULL)

noinst_HEADERS =                       \
    $(NULL)

libxcam_render_la_LIBTOOLFLAGS = --tag=disable-static
