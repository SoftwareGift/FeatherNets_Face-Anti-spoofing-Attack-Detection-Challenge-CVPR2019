/*
 * gstxcamsrc.c - gst xcamsrc plugin
 *
 *  Copyright (c) 2015 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Author: John Ye <john.ye@intel.com>
 */

/**
 * SECTION:element-xcamsrc
 *
 * FIXME:Describe xcamsrc here.
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch-1.0 xcamsrc sensor=0 capturemode=0x4000 memtype=4 buffercount=8 fpsn=25 fpsd=1   \
 *  width=1920 height=1080 pixelformat=0 field=0 bytesperline=3840 ! video/x-raw, format=NV12, \
 *  width=1920, height=1080, framerate=30/1 ! queue ! vaapiencode_h264 ! fakesink
 * ]|
 * </refsect2>
 */

#ifdef HAVE_CONFIG_H
#  include <config.h>
#endif

#include <gst/gst.h>
#include <gst/video/video.h>
#include <gst/video/video-format.h>
#include <linux/videodev2.h>

#include <stdio.h>
#include <signal.h>

#include "stub.h"
#include "fmt.h"
#include "gstxcamsrc.h"

GST_DEBUG_CATEGORY_STATIC (gst_xcamsrc_debug);
#define GST_CAT_DEFAULT gst_xcamsrc_debug

#define DEFAULT_BLOCKSIZE   1843200
#define DEFAULT_CAPTURE_DEVICE  "/dev/video3"

#define DEFAULT_PROP_SENSOR     0
#define DEFAULT_PROP_CAPTUREMODE    0
#define DEFAULT_PROP_MEMTYPE        1
#define DEFAULT_PROP_BUFFERCOUNT    6
#define DEFAULT_PROP_FPSN       0
#define DEFAULT_PROP_FPSD       0
#define DEFAULT_PROP_WIDTH      0
#define DEFAULT_PROP_HEIGHT     0
#define DEFAULT_PROP_PIXELFORMAT    0
#define DEFAULT_PROP_FIELD      0
#define DEFAULT_PROP_BYTESPERLINE   0

enum
{
    PROP_0,
    PROP_SENSOR,
    PROP_CAPTUREMODE,
    PROP_MEMTYPE,
    PROP_BUFFERCOUNT,
    PROP_FPSN,
    PROP_FPSD,
    PROP_WIDTH,
    PROP_HEIGHT,
    PROP_PIXELFORMAT,
    PROP_FIELD,
    PROP_BYTESPERLINE
};


#define gst_xcamsrc_parent_class parent_class
G_DEFINE_TYPE (Gstxcamsrc, gst_xcamsrc, GST_TYPE_PUSH_SRC);

GstCaps *gst_xcamsrc_get_all_caps (void);

static void gst_xcamsrc_finalize (GObject * object);
static void gst_xcamsrc_set_property (GObject *object, guint prop_id, const GValue *value, GParamSpec *pspec);
static void gst_xcamsrc_get_property (GObject *object, guint prop_id, GValue *value, GParamSpec *pspec);
static GstCaps* gst_xcamsrc_get_caps (GstBaseSrc *src, GstCaps *filter);
static gboolean gst_xcamsrc_set_caps (GstBaseSrc *src, GstCaps *caps);
static gboolean gst_xcamsrc_start (GstBaseSrc *src);
static gboolean gst_xcamsrc_stop (GstBaseSrc * basesrc);
static GstFlowReturn gst_xcamsrc_alloc (GstBaseSrc *src, guint64 offset, guint size, GstBuffer **buffer);
static GstFlowReturn gst_xcamsrc_fill (GstPushSrc *src, GstBuffer *out);

static void
gst_xcamsrc_class_init (GstxcamsrcClass * klass)
{
    GObjectClass *gobject_class;
    GstElementClass *element_class;
    GstBaseSrcClass *basesrc_class;
    GstPushSrcClass *pushsrc_class;

    gobject_class = (GObjectClass *) klass;
    element_class = (GstElementClass *) klass;
    basesrc_class = GST_BASE_SRC_CLASS (klass);
    pushsrc_class = GST_PUSH_SRC_CLASS (klass);

    gobject_class->finalize = gst_xcamsrc_finalize;
    gobject_class->set_property = gst_xcamsrc_set_property;
    gobject_class->get_property = gst_xcamsrc_get_property;

    g_object_class_install_property (gobject_class, PROP_SENSOR,
                                     g_param_spec_int ("sensor", "Sensor id", "Sensor id",
                                             0, G_MAXINT, DEFAULT_PROP_SENSOR, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

    g_object_class_install_property (gobject_class, PROP_CAPTUREMODE,
                                     g_param_spec_int ("capturemode", "capture mode", "capture mode",
                                             0, G_MAXINT, DEFAULT_PROP_CAPTUREMODE, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
    g_object_class_install_property (gobject_class, PROP_MEMTYPE,
                                     g_param_spec_int ("memtype", "memory type", "memory type",
                                             0, G_MAXINT, DEFAULT_PROP_MEMTYPE, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
    g_object_class_install_property (gobject_class, PROP_BUFFERCOUNT,
                                     g_param_spec_int ("buffercount", "buffer count", "buffer count",
                                             0, G_MAXINT, DEFAULT_PROP_BUFFERCOUNT, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
    g_object_class_install_property (gobject_class, PROP_FPSN,
                                     g_param_spec_int ("fpsn", "fps n", "fps n",
                                             0 , G_MAXINT, DEFAULT_PROP_FPSN, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
    g_object_class_install_property (gobject_class, PROP_FPSD,
                                     g_param_spec_int ("fpsd", "fps d", "fps d",
                                             0, G_MAXINT, DEFAULT_PROP_FPSD, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
    g_object_class_install_property (gobject_class, PROP_WIDTH,
                                     g_param_spec_int ("width", "width", "width",
                                             0, G_MAXINT, DEFAULT_PROP_WIDTH, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
    g_object_class_install_property (gobject_class, PROP_HEIGHT,
                                     g_param_spec_int ("height", "height", "height",
                                             0, G_MAXINT, DEFAULT_PROP_HEIGHT, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
    g_object_class_install_property (gobject_class, PROP_PIXELFORMAT,
                                     g_param_spec_int ("pixelformat", "pixelformat", "pixelformat",
                                             0, G_MAXINT, DEFAULT_PROP_PIXELFORMAT, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
    g_object_class_install_property (gobject_class, PROP_FIELD,
                                     g_param_spec_int ("field", "field", "field",
                                             0, G_MAXINT, DEFAULT_PROP_FIELD, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
    g_object_class_install_property (gobject_class, PROP_BYTESPERLINE,
                                     g_param_spec_int ("bytesperline", "bytes perline", "bytes perline",
                                             0, G_MAXINT, DEFAULT_PROP_BYTESPERLINE, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

    gst_element_class_set_details_simple(element_class,
                                         "Libxcam Source",
                                         "Source/Base",
                                         "Capture camera video using xcam library",
                                         "John Ye <john.ye@intel.com>");

    gst_element_class_add_pad_template (element_class,
                                        gst_pad_template_new ("src", GST_PAD_SRC, GST_PAD_ALWAYS, gst_xcamsrc_get_all_caps ()));

    basesrc_class->get_caps = GST_DEBUG_FUNCPTR (gst_xcamsrc_get_caps);
    basesrc_class->set_caps = GST_DEBUG_FUNCPTR (gst_xcamsrc_set_caps);
    basesrc_class->alloc = GST_DEBUG_FUNCPTR (gst_xcamsrc_alloc);

    basesrc_class->start = GST_DEBUG_FUNCPTR (gst_xcamsrc_start);
    basesrc_class->stop = GST_DEBUG_FUNCPTR (gst_xcamsrc_stop);
    pushsrc_class->fill = GST_DEBUG_FUNCPTR (gst_xcamsrc_fill);
}

Gstxcamsrc *g_src;

void handler (int sig)
{
    libxcam_stop ();
    libxcam_close ();
    exit (1);
}

static void
gst_xcamsrc_init (Gstxcamsrc *xcamsrc)
{
    g_src = xcamsrc;
    signal (SIGSEGV, handler);
    libxcam_set_device_name (DEFAULT_CAPTURE_DEVICE);
    gst_base_src_set_format (GST_BASE_SRC (xcamsrc), GST_FORMAT_TIME);
    gst_base_src_set_live (GST_BASE_SRC (xcamsrc), TRUE);

    xcamsrc->_fps_n = 0;
    xcamsrc->_fps_d = 0;

    gst_base_src_set_blocksize (GST_BASE_SRC (xcamsrc), DEFAULT_BLOCKSIZE);
}

static void
gst_xcamsrc_finalize (GObject * object)
{
    Gstxcamsrc *src = GST_XCAMSRC (object);

    libxcam_stop ();
    libxcam_close ();

    G_OBJECT_CLASS (parent_class)->finalize (object);
}

static gboolean
gst_xcamsrc_start (GstBaseSrc *src)
{
    Gstxcamsrc *xcamsrc = GST_XCAMSRC (src);

    xcamsrc->offset = 0;
    xcamsrc->ctrl_time = 0;

    gst_object_sync_values (GST_OBJECT (src), xcamsrc->ctrl_time);

    return TRUE;
}

static gboolean
gst_xcamsrc_stop (GstBaseSrc * basesrc)
{
    Gstxcamsrc *src = GST_XCAMSRC_CAST (basesrc);
    libxcam_stop ();
    return TRUE;
}

static GstCaps*
gst_xcamsrc_get_caps (GstBaseSrc *src, GstCaps *filter)
{
    Gstxcamsrc *xcamsrc = GST_XCAMSRC (src);
    return gst_pad_get_pad_template_caps (GST_BASE_SRC_PAD (xcamsrc));
}

static gboolean
gst_xcamsrc_set_caps (GstBaseSrc *src, GstCaps *caps)
{
    Gstxcamsrc *xcamsrc = GST_XCAMSRC (src);

    guint32 block_size = DEFAULT_BLOCKSIZE;
    /**
     * set_sensor_id
     * set_capture_mode
     * set_mem_type
     * set_buffer_count
     * set_framerate
     * open
     * set_format
     *
     **/
    libxcam_set_framerate (xcamsrc->_fps_n, xcamsrc->_fps_d);
    libxcam_open ();
    libxcam_set_format (xcamsrc->width, xcamsrc->height, xcamsrc->pixelformat, xcamsrc->field, xcamsrc->bytes_perline);

    libxcam_get_blocksize (&block_size);
    gst_base_src_set_blocksize (GST_BASE_SRC (xcamsrc), block_size);

    xcamsrc->duration = gst_util_uint64_scale_int (GST_SECOND, xcamsrc->_fps_d, xcamsrc->_fps_n);
    xcamsrc->pool = gst_xcambufferpool_new (src, caps);

    gst_buffer_pool_set_active (GST_BUFFER_POOL_CAST (xcamsrc->pool), TRUE);
    return TRUE;
}



static GstFlowReturn gst_xcamsrc_alloc (GstBaseSrc *src, guint64 offset, guint size, GstBuffer **buffer)
{
    GstFlowReturn ret;
    Gstxcamsrc *xcamsrc = GST_XCAMSRC (src);

    ret = gst_buffer_pool_acquire_buffer (xcamsrc->pool, buffer, NULL);
    return ret;
}

// FIXME: timestamp is already set in xcore, isn't it?
static GstFlowReturn
gst_xcamsrc_fill (GstPushSrc *basesrc, GstBuffer *buf)
{
    Gstxcamsrc *src = GST_XCAMSRC_CAST (basesrc);
    GstClockTime abs_time, base_time, timestamp, duration;
    GstClock *clock;
    GstClockTime delay;

    timestamp = GST_BUFFER_TIMESTAMP (buf);
    duration = src->duration;

    GST_OBJECT_LOCK (src);
    if ((clock = GST_ELEMENT_CLOCK (src))) {
        base_time = GST_ELEMENT (src)->base_time;
        gst_object_ref (clock);
    } else {
        base_time = GST_CLOCK_TIME_NONE;
    }
    GST_OBJECT_UNLOCK (src);

    if (clock) {
        abs_time = gst_clock_get_time (clock);
        gst_object_unref (clock);
    } else {
        abs_time = GST_CLOCK_TIME_NONE;
    }

    if (timestamp != GST_CLOCK_TIME_NONE) {
        struct timespec now;
        GstClockTime gstnow;

        clock_gettime (CLOCK_MONOTONIC, &now);
        gstnow = GST_TIMESPEC_TO_TIME (now);

        if (gstnow < timestamp && (timestamp - gstnow) > (10 * GST_SECOND)) {
            GTimeVal now;

            g_get_current_time (&now);
            gstnow = GST_TIMEVAL_TO_TIME (now);
        }
        if (gstnow > timestamp) {
            delay = gstnow - timestamp;
        } else {
            delay = 0;
        }
    } else {
        if (GST_CLOCK_TIME_IS_VALID (duration))
            delay = duration;
        else
            delay = 0;
    }

    GST_BUFFER_OFFSET (buf) = src->offset++;
    GST_BUFFER_OFFSET_END (buf) = src->offset;

    if (G_LIKELY (abs_time != GST_CLOCK_TIME_NONE)) {
        timestamp = abs_time - base_time;

        if (timestamp > delay)
            timestamp -= delay;
        else
            timestamp = 0;
    } else {
        timestamp = GST_CLOCK_TIME_NONE;
    }

    if (GST_CLOCK_TIME_IS_VALID (duration)) {
        src->ctrl_time += duration;
    } else {
        src->ctrl_time = timestamp;
    }

    gst_object_sync_values (GST_OBJECT (src), src->ctrl_time);

    GST_BUFFER_TIMESTAMP (buf) = timestamp;
    GST_BUFFER_DURATION (buf) = duration;

    return GST_FLOW_OK;
}

static void gst_xcamsrc_get_property (GObject *object,
                                      guint prop_id, GValue *value, GParamSpec *pspec)
{
    //TODO
}

// FIXME do we need to support all these properties?
static void gst_xcamsrc_set_property (GObject *object,
                                      guint prop_id, const GValue *value, GParamSpec *pspec)
{
    Gstxcamsrc *src = GST_XCAMSRC (object);
    int val;
    enum v4l2_memory set_val;

    switch (prop_id) {
    case PROP_SENSOR:
        libxcam_set_sensor_id (g_value_get_int (value));
        break;
    case PROP_CAPTUREMODE:
        libxcam_set_capture_mode (g_value_get_int (value));
        break;
    case PROP_MEMTYPE:
        val = g_value_get_int (value);
        if (val == 1)
            set_val = V4L2_MEMORY_MMAP;
        else if (val == 2)
            set_val = V4L2_MEMORY_USERPTR;
        else if (val == 3)
            set_val = V4L2_MEMORY_OVERLAY;
        else
            set_val = V4L2_MEMORY_DMABUF;
        libxcam_set_mem_type (set_val);
        break;
    case PROP_BUFFERCOUNT:
        src->buf_count = g_value_get_int (value);
        libxcam_set_buffer_count (g_value_get_int (value));
        break;
    case PROP_FPSN:
        src->_fps_n = g_value_get_int (value);
        break;
    case PROP_FPSD:
        src->_fps_d = g_value_get_int (value);
        break;
    case PROP_WIDTH:
        src->width = g_value_get_int (value);
        break;
    case PROP_HEIGHT:
        src->height = g_value_get_int (value);
        break;
    case PROP_PIXELFORMAT:
        // FIXME
        src->pixelformat = V4L2_PIX_FMT_NV12;
        break;
    case PROP_FIELD:
        // TODO
        src->field = V4L2_FIELD_NONE;
        break;
    case PROP_BYTESPERLINE:
        src->bytes_perline = g_value_get_int (value);
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
        break;
    }
}

GstCaps *
gst_xcamsrc_get_all_caps (void)
{
    static GstCaps *caps = NULL;
    if (caps == NULL) {
        caps = gst_caps_new_empty ();
        caps_append(caps);
    }
    return gst_caps_ref (caps);
}


static gboolean
xcamsrc_init (GstPlugin * xcamsrc)
{
    GST_DEBUG_CATEGORY_INIT (gst_xcamsrc_debug, "xcamsrc",
                             0, "xcamsrc");

    return gst_element_register (xcamsrc, "xcamsrc", GST_RANK_NONE,
                                 GST_TYPE_XCAMSRC);
}

#ifndef PACKAGE
#define PACKAGE "libxam"
#endif

GST_PLUGIN_DEFINE (
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    xcamsrc,
    "xcamsrc",
    xcamsrc_init,
    VERSION,
    GST_LICENSE_UNKNOWN,
    "Xcamsrc",
    "https://github.com/01org/libxcam"
)
