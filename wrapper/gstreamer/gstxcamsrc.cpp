/*
 * gstxcamsrc.cpp - gst xcamsrc plugin
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

#include "gstxcamsrc.h"
#include "v4l2dev.h"
#include "stub.h"
#include "fmt.h"

using namespace XCam;

XCAM_BEGIN_DECLARE

#ifdef HAVE_CONFIG_H
#  include <config.h>
#endif

#include <gst/gst.h>
#include <gst/video/video.h>
#include <gst/video/video-format.h>
#include <linux/videodev2.h>

#include <stdio.h>
#include <signal.h>

#include "gstxcamsrc.h"
#include "gstxcaminterface.h"


GST_DEBUG_CATEGORY_STATIC (gst_xcamsrc_debug);
#define GST_CAT_DEFAULT gst_xcamsrc_debug

#define GST_TYPE_XCAM_SRC_IMAGE_PROCESSOR (gst_xcam_src_image_processor_get_type ())
static GType
gst_xcam_src_image_processor_get_type (void)
{
    static GType g_type = 0;
    static const GEnumValue image_processor_types[] = {
        {ISP_IMAGE_PROCESSOR, "ISP image processor", "isp"},
        {CL_IMAGE_PROCESSOR, "CL image processor", "cl"},
        {0, NULL, NULL},
    };

    if (g_once_init_enter (&g_type)) {
        const GType type =
            g_enum_register_static ("GstXcamSrcImageProcessor", image_processor_types);
        g_once_init_leave (&g_type, type);
    }

    return g_type;
}

#define GST_TYPE_XCAM_SRC_ANALYZER (gst_xcam_src_analyzer_get_type ())
static GType
gst_xcam_src_analyzer_get_type (void)
{
    static GType g_type = 0;
    static const GEnumValue analyzer_types[] = {
        {SIMPLE_ANALYZER, "simple 3A analyzer", "simple"},
        {AIQ_ANALYZER, "aiq 3A analyzer", "aiq"},
        {0, NULL, NULL},
    };

    if (g_once_init_enter (&g_type)) {
        const GType type =
            g_enum_register_static ("GstXcamSrcAnalyzer", analyzer_types);
        g_once_init_leave (&g_type, type);
    }

    return g_type;
}

enum
{
    PROP_0,
    PROP_DEVICE,
    PROP_COLOREFFECT,
    PROP_SENSOR,
    PROP_CAPTURE_MODE,
    PROP_ENABLE_3A,
    PROP_IO_MODE,
    PROP_BUFFERCOUNT,
    PROP_FPSN,
    PROP_FPSD,
    PROP_WIDTH,
    PROP_HEIGHT,
    PROP_PIXELFORMAT,
    PROP_FIELD,
    PROP_BYTESPERLINE,
    PROP_IMAGE_PROCESSOR,
    PROP_3A_ANALYZER
};

static void gst_xcamsrc_xcam_3a_interface_init (GstXCam3AInterface *iface);

#define gst_xcamsrc_parent_class parent_class
G_DEFINE_TYPE_WITH_CODE  (Gstxcamsrc, gst_xcamsrc, GST_TYPE_PUSH_SRC,
                          G_IMPLEMENT_INTERFACE (GST_TYPE_XCAM_3A_IF,
                                  gst_xcamsrc_xcam_3a_interface_init));

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

static gboolean xcamsrc_init (GstPlugin * xcamsrc);

XCAM_END_DECLARE

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

    g_object_class_install_property (gobject_class, PROP_DEVICE,
                                     g_param_spec_string ("device", "Device", "Device location",
                                             DEFAULT_PROP_DEVICE_NAME, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property (gobject_class, PROP_COLOREFFECT,
                                     g_param_spec_int ("coloreffect", "color effect", "0, none\t 1,sky blue\t 2,skin whiten low\t 3,skin whiten\t 4,skin whiten hight\t 5,sepia",
                                             G_MININT, G_MAXINT, 0, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_CONTROLLABLE)));

    g_object_class_install_property (gobject_class, PROP_ENABLE_3A,
                                     g_param_spec_boolean ("enable-3a", "Enable 3A", "Enable 3A",
                                             true, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property (gobject_class, PROP_SENSOR,
                                     g_param_spec_int ("sensor-id", "Sensor id", "Sensor id to input",
                                             0, G_MAXINT, DEFAULT_PROP_SENSOR, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS) ));

    g_object_class_install_property (gobject_class, PROP_CAPTURE_MODE,
                                     g_param_spec_int ("capture-mode", "capture mode", "capture mode",
                                             0, G_MAXINT, DEFAULT_PROP_CAPTURE_MODE, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS) ));
    g_object_class_install_property (gobject_class, PROP_IO_MODE,
                                     g_param_spec_int ("io-mode", "IO mode", "I/O mode",
                                             0, G_MAXINT, DEFAULT_PROP_IO_MODE, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS) ));
    g_object_class_install_property (gobject_class, PROP_BUFFERCOUNT,
                                     g_param_spec_int ("buffercount", "buffer count", "buffer count",
                                             0, G_MAXINT, DEFAULT_PROP_BUFFERCOUNT, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS) ));
    g_object_class_install_property (gobject_class, PROP_FPSN,
                                     g_param_spec_int ("fpsn", "fps n", "fps n",
                                             0 , G_MAXINT, DEFAULT_PROP_FPSN, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS) ));
    g_object_class_install_property (gobject_class, PROP_FPSD,
                                     g_param_spec_int ("fpsd", "fps d", "fps d",
                                             0, G_MAXINT, DEFAULT_PROP_FPSD, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS) ));
    g_object_class_install_property (gobject_class, PROP_WIDTH,
                                     g_param_spec_int ("width", "width", "width",
                                             0, G_MAXINT, DEFAULT_PROP_WIDTH, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS) ));
    g_object_class_install_property (gobject_class, PROP_HEIGHT,
                                     g_param_spec_int ("height", "height", "height",
                                             0, G_MAXINT, DEFAULT_PROP_HEIGHT, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS) ));
    g_object_class_install_property (gobject_class, PROP_PIXELFORMAT,
                                     g_param_spec_int ("pixelformat", "pixelformat", "pixelformat",
                                             0, G_MAXINT, DEFAULT_PROP_PIXELFORMAT, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS) ));
    g_object_class_install_property (gobject_class, PROP_FIELD,
                                     g_param_spec_int ("field", "field", "field",
                                             0, G_MAXINT, DEFAULT_PROP_FIELD, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS) ));
    g_object_class_install_property (gobject_class, PROP_BYTESPERLINE,
                                     g_param_spec_int ("bytesperline", "bytes perline", "bytes perline",
                                             0, G_MAXINT, DEFAULT_PROP_BYTESPERLINE, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS) ));
    g_object_class_install_property (gobject_class, PROP_IMAGE_PROCESSOR,
                                     g_param_spec_enum ("imageprocessor", "image processor", "image processor",
                                             GST_TYPE_XCAM_SRC_IMAGE_PROCESSOR, ISP_IMAGE_PROCESSOR,
                                             (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
    g_object_class_install_property (gobject_class, PROP_3A_ANALYZER,
                                     g_param_spec_enum ("analyzer", "3a analyzer", "3a analyzer",
                                             GST_TYPE_XCAM_SRC_ANALYZER, SIMPLE_ANALYZER,
                                             (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

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

// FIXME remove this function?
static void
gst_xcamsrc_init (Gstxcamsrc *xcamsrc)
{
    gst_base_src_set_format (GST_BASE_SRC (xcamsrc), GST_FORMAT_TIME);
    gst_base_src_set_live (GST_BASE_SRC (xcamsrc), TRUE);

    xcamsrc->buf_count = DEFAULT_PROP_BUFFERCOUNT; //8
    xcamsrc->_fps_n = DEFAULT_PROP_FPSN; //25
    xcamsrc->_fps_d = DEFAULT_PROP_FPSD; //1
    xcamsrc->width = DEFAULT_PROP_WIDTH; //1920
    xcamsrc->height = DEFAULT_PROP_HEIGHT; //1080
    xcamsrc->pixelformat = V4L2_PIX_FMT_NV12;
    xcamsrc->field = V4L2_FIELD_NONE; //0
    xcamsrc->bytes_perline = DEFAULT_PROP_BYTESPERLINE; // 3840
    xcamsrc->image_processor_type = ISP_IMAGE_PROCESSOR;
    xcamsrc->analyzer_type = SIMPLE_ANALYZER;

    gst_base_src_set_blocksize (GST_BASE_SRC (xcamsrc), DEFAULT_BLOCKSIZE);
}

static void
gst_xcamsrc_finalize (GObject * object)
{
    G_OBJECT_CLASS (parent_class)->finalize (object);
}

static gboolean
gst_xcamsrc_start (GstBaseSrc *src)
{
    Gstxcamsrc *xcamsrc = GST_XCAMSRC (src);

    xcamsrc->offset = 0;
    xcamsrc->ctrl_time = 0;

    gst_object_sync_values (GST_OBJECT (src), xcamsrc->ctrl_time);

    SmartPtr<MainDeviceManager> device_manager = DeviceManagerInstance::device_manager_instance();

    SmartPtr<V4l2Device> capture_device;
    if (xcamsrc->capture_mode == V4L2_CAPTURE_MODE_STILL)
        capture_device = new AtomispDevice (CAPTURE_DEVICE_STILL);
    else
        capture_device = new AtomispDevice (CAPTURE_DEVICE_VIDEO);
    device_manager->set_capture_device (capture_device);
    capture_device->set_sensor_id (xcamsrc->sensor_id);
    capture_device->set_capture_mode (xcamsrc->capture_mode);
    capture_device->set_mem_type (xcamsrc->mem_type);
    capture_device->set_buffer_count (xcamsrc->buf_count);
    capture_device->set_framerate (xcamsrc->_fps_n, xcamsrc->_fps_d);
    capture_device->open ();
    capture_device->set_format (xcamsrc->width, xcamsrc->height, xcamsrc->pixelformat, xcamsrc->field, xcamsrc->bytes_perline);

    SmartPtr<V4l2SubDevice> event_device = new V4l2SubDevice (DEFAULT_EVENT_DEVICE);
    device_manager->set_event_device (event_device);
    event_device->open ();
    event_device->subscribe_event (V4L2_EVENT_ATOMISP_3A_STATS_READY);
    event_device->subscribe_event (V4L2_EVENT_FRAME_SYNC);

    SmartPtr<IspController> isp_controller = new IspController (capture_device);
    device_manager->set_isp_controller (isp_controller);

    SmartPtr<ImageProcessor> isp_processor;
#if HAVE_LIBCL
    SmartPtr<CL3aImageProcessor> cl_processor = new CL3aImageProcessor ();
#endif
    switch (xcamsrc->image_processor_type) {
#if HAVE_LIBCL
    case CL_IMAGE_PROCESSOR:
        isp_processor = new IspExposureImageProcessor (isp_controller);
        XCAM_ASSERT (isp_processor.ptr ());
        device_manager->add_image_processor (isp_processor);
        cl_processor = new CL3aImageProcessor ();
        cl_processor->set_stats_callback (device_manager);
        device_manager->add_image_processor (cl_processor);
        device_manager->set_cl_image_processor (cl_processor);
        break;
#endif
    default:
        isp_processor = new IspImageProcessor (isp_controller);
        device_manager->add_image_processor (isp_processor);
    }

    SmartPtr<X3aAnalyzer> analyzer;
    switch (xcamsrc->analyzer_type) {
#if HAVE_IA_AIQ
    case AIQ_ANALYZER:
        analyzer = new X3aAnalyzerAiq (isp_controller, DEFAULT_CPF_FILE_NAME);
        break;
#endif
    default:
        analyzer = new X3aAnalyzerSimple ();
    }
    device_manager->set_analyzer (analyzer);

    device_manager->start ();

    return TRUE;
}

static gboolean
gst_xcamsrc_stop (GstBaseSrc * basesrc)
{
    Gstxcamsrc *src = GST_XCAMSRC_CAST (basesrc);
    SmartPtr<MainDeviceManager> device_manager = DeviceManagerInstance::device_manager_instance();

    device_manager->stop();
    device_manager->get_capture_device()->close ();
    device_manager->get_event_device()->close ();
    return TRUE;
}

static GstCaps*
gst_xcamsrc_get_caps (GstBaseSrc *src, GstCaps *filter)
{
    Gstxcamsrc *xcamsrc = GST_XCAMSRC (src);
    return gst_pad_get_pad_template_caps (GST_BASE_SRC_PAD (xcamsrc));
}

extern "C" GstBufferPool *
gst_xcambufferpool_new (Gstxcamsrc *xcamsrc, GstCaps *caps);

static gboolean
gst_xcamsrc_set_caps (GstBaseSrc *src, GstCaps *caps)
{
    Gstxcamsrc *xcamsrc = GST_XCAMSRC (src);

    guint32 block_size = DEFAULT_BLOCKSIZE;
    SmartPtr<MainDeviceManager> device_manager = DeviceManagerInstance::device_manager_instance();
    SmartPtr<V4l2Device> device = device_manager->get_capture_device();


    struct v4l2_format format;
    device->get_format (format);
    block_size = format.fmt.pix.sizeimage;

    gst_base_src_set_blocksize (GST_BASE_SRC (xcamsrc), block_size);

    xcamsrc->duration = gst_util_uint64_scale_int (GST_SECOND, xcamsrc->_fps_d, xcamsrc->_fps_n);
    xcamsrc->pool = gst_xcambufferpool_new ((Gstxcamsrc*)src, caps);

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
    case PROP_DEVICE:
    case PROP_ENABLE_3A:
    case PROP_COLOREFFECT:
        break;
    case PROP_SENSOR:
        src->sensor_id = g_value_get_int (value);
        break;
    case PROP_CAPTURE_MODE:
        switch (g_value_get_int (value)) {
        case 0:
            src->capture_mode = V4L2_CAPTURE_MODE_STILL;
            break;
        case 1:
            src->capture_mode = V4L2_CAPTURE_MODE_VIDEO;
            break;
        case 2:
            src->capture_mode = V4L2_CAPTURE_MODE_PREVIEW;
            break;
        default:
            XCAM_LOG_ERROR ("Invalid capure mode");
            break;
        }
        break;
    case PROP_IO_MODE:
        switch (g_value_get_int (value)) {
        case 1:
            src->mem_type = V4L2_MEMORY_MMAP;
            break;
        case 2:
            src->mem_type = V4L2_MEMORY_USERPTR;
            break;
        case 3:
            src->mem_type = V4L2_MEMORY_OVERLAY;
            break;
        case 4:
            src->mem_type = V4L2_MEMORY_DMABUF;
            break;
        default:
            XCAM_LOG_ERROR ("Invalid io mode");
            break;
        }
        break;
    case PROP_BUFFERCOUNT:
        src->buf_count = g_value_get_int (value);
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
    case PROP_IMAGE_PROCESSOR:
        src->image_processor_type = (ImageProcessorType)g_value_get_enum (value);
        if (src->image_processor_type == ISP_IMAGE_PROCESSOR) {
            src->capture_mode = V4L2_CAPTURE_MODE_VIDEO;
            src->pixelformat = V4L2_PIX_FMT_NV12;
        }
        else if (src->image_processor_type == CL_IMAGE_PROCESSOR) {
            src->capture_mode = V4L2_CAPTURE_MODE_STILL;
            src->pixelformat = V4L2_PIX_FMT_SGRBG10;
        }
        break;
    case PROP_3A_ANALYZER:
        src->analyzer_type = (AnalyzerType)g_value_get_enum (value);
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

static void
gst_xcamsrc_xcam_3a_interface_init (GstXCam3AInterface *iface)
{
    iface->set_white_balance_mode = gst_xcamsrc_set_white_balance_mode;
    iface->set_awb_speed = gst_xcamsrc_set_awb_speed;

    iface->set_wb_color_temperature_range = gst_xcamsrc_set_wb_color_temperature_range;
    iface->set_manual_wb_gain = gst_xcamsrc_set_manual_wb_gain;

    iface->set_exposure_mode = gst_xcamsrc_set_exposure_mode;
    iface->set_ae_metering_mode = gst_xcamsrc_set_ae_metering_mode;
    iface->set_exposure_window = gst_xcamsrc_set_exposure_window;
    iface->set_exposure_value_offset = gst_xcamsrc_set_exposure_value_offset;
    iface->set_ae_speed = gst_xcamsrc_set_ae_speed;

    iface->set_exposure_flicker_mode = gst_xcamsrc_set_exposure_flicker_mode;
    iface->get_exposure_flicker_mode = gst_xcamsrc_get_exposure_flicker_mode;
    iface->get_current_exposure_time = gst_xcamsrc_get_current_exposure_time;
    iface->get_current_analog_gain = gst_xcamsrc_get_current_analog_gain;
    iface->set_manual_exposure_time = gst_xcamsrc_set_manual_exposure_time;
    iface->set_manual_analog_gain = gst_xcamsrc_set_manual_analog_gain;
    iface->set_aperture = gst_xcamsrc_set_aperture;
    iface->set_max_analog_gain = gst_xcamsrc_set_max_analog_gain;
    iface->get_max_analog_gain = gst_xcamsrc_get_max_analog_gain;
    iface->set_exposure_time_range = gst_xcamsrc_set_exposure_time_range;
    iface->get_exposure_time_range = gst_xcamsrc_get_exposure_time_range;
    iface->set_dvs = gst_xcamsrc_set_dvs;
    iface->set_noise_reduction_level = gst_xcamsrc_set_noise_reduction_level;
    iface->set_temporal_noise_reduction_level = gst_xcamsrc_set_temporal_noise_reduction_level;
    iface->set_gamma_table = gst_xcamsrc_set_gamma_table;
    iface->set_gbce = gst_xcamsrc_set_gbce;
    iface->set_manual_brightness = gst_xcamsrc_set_manual_brightness;
    iface->set_manual_contrast = gst_xcamsrc_set_manual_contrast;
    iface->set_manual_hue = gst_xcamsrc_set_manual_hue;
    iface->set_manual_saturation = gst_xcamsrc_set_manual_saturation;
    iface->set_manual_sharpness = gst_xcamsrc_set_manual_sharpness;
    iface->set_night_mode = gst_xcamsrc_set_night_mode;
    iface->set_hdr_mode = gst_xcamsrc_set_hdr_mode;
    iface->set_denoise_mode = gst_xcamsrc_set_denoise_mode;
    iface->set_gamma_mode = gst_xcamsrc_set_gamma_mode;
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
