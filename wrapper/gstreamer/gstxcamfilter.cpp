/*
 * gstxcamfilter.cpp -gst xcamfilter plugin
 *
 *  Copyright (c) 2016 Intel Corporation
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
 * Author: Yinhang Liu <yinhangx.liu@intel.com>
 */

#include "gstxcamfilter.h"
#include "gstxcambuffermeta.h"

#include <gst/gstmeta.h>
#include <gst/allocators/gstdmabuf.h>

using namespace XCam;
using namespace GstXCam;

#define DEFAULT_SMART_ANALYSIS_LIB_DIR      "/usr/lib/xcam/plugins/smart"
#define DEFAULT_DELAY_BUFFER_NUM            2

#define DEFAULT_PROP_BUFFERCOUNT            8
#define DEFAULT_PROP_COPY_MODE              COPY_MODE_CPU
#define DEFAULT_PROP_DEFOG_MODE             DEFOG_NONE
#define DEFAULT_PROP_WAVELET_MODE           NONE_WAVELET
#define DEFAULT_PROP_3D_DENOISE_MODE        DENOISE_3D_NONE
#define DEFAULT_PROP_ENABLE_WIREFRAME       FALSE
#define DEFAULT_PROP_ENABLE_IMAGE_WARP      FALSE
#define DEFAULT_PROP_ENABLE_IMAGE_STITCH    FALSE
#define DEFAULT_PROP_STITCH_ENABLE_SEAM     FALSE
#define DEFAULT_PROP_STITCH_SCALE_MODE      CLBlenderScaleLocal
#define DEFAULT_PROP_STITCH_FISHEYE_MAP     FALSE
#define DEFAULT_PROP_STITCH_LSC             FALSE
#define DEFAULT_PROP_STITCH_FM_OCL          FALSE
#define DEFAULT_PROP_STITCH_RES_MODE        StitchRes1080P

XCAM_BEGIN_DECLARE

enum {
    PROP_0,
    PROP_BUFFERCOUNT,
    PROP_COPY_MODE,
    PROP_DEFOG_MODE,
    PROP_WAVELET_MODE,
    PROP_DENOISE_3D_MODE,
    PROP_ENABLE_WIREFRAME,
    PROP_ENABLE_IMAGE_WARP,
    PROP_ENABLE_IMAGE_STITCH,
    PROP_STITCH_ENABLE_SEAM,
    PROP_STITCH_SCALE_MODE,
    PROP_STITCH_FISHEYE_MAP,
    PROP_STITCH_LSC,
    PROP_STITCH_FM_OCL,
    PROP_STITCH_RES_MODE
};

#define GST_TYPE_XCAM_FILTER_COPY_MODE (gst_xcam_filter_copy_mode_get_type ())
static GType
gst_xcam_filter_copy_mode_get_type (void)
{
    static GType g_type = 0;
    static const GEnumValue copy_mode_types[] = {
        {COPY_MODE_CPU, "Copy buffer with CPU", "cpu"},
        {COPY_MODE_DMA, "Copy buffer with DMA", "dma"},
        {0, NULL, NULL}
    };

    if (g_once_init_enter (&g_type)) {
        const GType type =
            g_enum_register_static ("GstXCamFilterCopyModeType", copy_mode_types);
        g_once_init_leave (&g_type, type);
    }

    return g_type;
}

#define GST_TYPE_XCAM_FILTER_DEFOG_MODE (gst_xcam_filter_defog_mode_get_type ())
static GType
gst_xcam_filter_defog_mode_get_type (void)
{
    static GType g_type = 0;
    static const GEnumValue defog_mode_types [] = {
        {DEFOG_NONE, "Defog disabled", "none"},
        {DEFOG_RETINEX, "Defog retinex", "retinex"},
        {DEFOG_DCP, "Defog dark channel prior", "dcp"},
        {0, NULL, NULL}
    };

    if (g_once_init_enter (&g_type)) {
        const GType type =
            g_enum_register_static ("GstXCamFilterDefogModeType", defog_mode_types);
        g_once_init_leave (&g_type, type);
    }

    return g_type;
}

#define GST_TYPE_XCAM_FILTER_WAVELET_MODE (gst_xcam_filter_wavelet_mode_get_type ())
static GType
gst_xcam_filter_wavelet_mode_get_type (void)
{
    static GType g_type = 0;
    static const GEnumValue wavelet_mode_types[] = {
        {NONE_WAVELET, "Wavelet disabled", "none"},
        {HAT_WAVELET_Y, "Hat wavelet Y", "hat Y"},
        {HAT_WAVELET_UV, "Hat wavelet UV", "hat UV"},
        {HARR_WAVELET_Y, "Haar wavelet Y", "haar Y"},
        {HARR_WAVELET_UV, "Haar wavelet UV", "haar UV"},
        {HARR_WAVELET_YUV, "Haar wavelet YUV", "haar YUV"},
        {HARR_WAVELET_BAYES, "Haar wavelet bayes shrink", "haar Bayes"},
        {0, NULL, NULL},
    };

    if (g_once_init_enter (&g_type)) {
        const GType type =
            g_enum_register_static ("GstXCamFilterWaveletModeType", wavelet_mode_types);
        g_once_init_leave (&g_type, type);
    }

    return g_type;
}

#define GST_TYPE_XCAM_FILTER_3D_DENOISE_MODE (gst_xcam_filter_3d_denoise_mode_get_type ())
static GType
gst_xcam_filter_3d_denoise_mode_get_type (void)
{
    static GType g_type = 0;
    static const GEnumValue denoise_3d_mode_types [] = {
        {DENOISE_3D_NONE, "3D Denoise disabled", "none"},
        {DENOISE_3D_YUV, "3D Denoise yuv", "yuv"},
        {DENOISE_3D_UV, "3D Denoise uv", "uv"},
        {0, NULL, NULL}
    };

    if (g_once_init_enter (&g_type)) {
        const GType type =
            g_enum_register_static ("GstXCamFilter3DDenoiseModeType", denoise_3d_mode_types);
        g_once_init_leave (&g_type, type);
    }

    return g_type;
}

#define GST_TYPE_XCAM_FILTER_STITCH_SCALE_MODE (gst_xcam_filter_stitch_scale_mode_get_type ())
static GType
gst_xcam_filter_stitch_scale_mode_get_type (void)
{
    static GType g_type = 0;
    static const GEnumValue stitch_scale_mode_types [] = {
        {CLBlenderScaleLocal, "Image stitch local scale", "local"},
        {CLBlenderScaleGlobal, "Image stitch glocal scale", "global"},
        {0, NULL, NULL}
    };

    if (g_once_init_enter (&g_type)) {
        const GType type =
            g_enum_register_static ("GstXCamFilterStitchScaleModeType", stitch_scale_mode_types);
        g_once_init_leave (&g_type, type);
    }

    return g_type;
}

#define GST_TYPE_XCAM_FILTER_STITCH_RES_MODE (gst_xcam_filter_stitch_res_mode_get_type ())
static GType
gst_xcam_filter_stitch_res_mode_get_type (void)
{
    static GType g_type = 0;
    static const GEnumValue stitch_res_mode_types [] = {
        {StitchRes1080P, "Image stitch 1080P mode", "1080p"},
        {StitchRes4K, "Image stitch 4K mode", "4k"},
        {0, NULL, NULL}
    };

    if (g_once_init_enter (&g_type)) {
        const GType type =
            g_enum_register_static ("GstXCamFilterStitchResModeType", stitch_res_mode_types);
        g_once_init_leave (&g_type, type);
    }

    return g_type;
}

static GstStaticPadTemplate gst_xcam_sink_factory =
    GST_STATIC_PAD_TEMPLATE ("sink",
                             GST_PAD_SINK,
                             GST_PAD_ALWAYS,
                             GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE ("{ NV12 }")));

static GstStaticPadTemplate gst_xcam_src_factory =
    GST_STATIC_PAD_TEMPLATE ("src",
                             GST_PAD_SRC,
                             GST_PAD_ALWAYS,
                             GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE ("{ NV12 }")));

GST_DEBUG_CATEGORY (gst_xcam_filter_debug);
#define GST_CAT_DEFAULT gst_xcam_filter_debug

#define gst_xcam_filter_parent_class parent_class
G_DEFINE_TYPE (GstXCamFilter, gst_xcam_filter, GST_TYPE_BASE_TRANSFORM);

static void gst_xcam_filter_finalize (GObject * object);
static void gst_xcam_filter_set_property (GObject *object, guint prop_id, const GValue *value, GParamSpec *pspec);
static void gst_xcam_filter_get_property (GObject *object, guint prop_id, GValue *value, GParamSpec *pspec);
static gboolean gst_xcam_filter_start (GstBaseTransform *trans);
static GstCaps *gst_xcam_filter_transform_caps (
    GstBaseTransform *trans, GstPadDirection direction, GstCaps *caps, GstCaps *filter);
static gboolean gst_xcam_filter_set_caps (GstBaseTransform *trans, GstCaps *incaps, GstCaps *outcaps);
static gboolean gst_xcam_filter_stop (GstBaseTransform *trans);
static void gst_xcam_filter_before_transform (GstBaseTransform *trans, GstBuffer *buffer);
static GstFlowReturn gst_xcam_filter_prepare_output_buffer (GstBaseTransform * trans, GstBuffer *input, GstBuffer **outbuf);
static GstFlowReturn gst_xcam_filter_transform (GstBaseTransform *trans, GstBuffer *inbuf, GstBuffer *outbuf);

XCAM_END_DECLARE

static void
gst_xcam_filter_class_init (GstXCamFilterClass *class_self)
{
    GObjectClass *gobject_class;
    GstElementClass *element_class;
    GstBaseTransformClass *basetrans_class;

    gobject_class = (GObjectClass *) class_self;
    element_class = (GstElementClass *) class_self;
    basetrans_class = (GstBaseTransformClass *) class_self;

    GST_DEBUG_CATEGORY_INIT (gst_xcam_filter_debug, "xcamfilter", 0, "LibXCam filter plugin");

    gobject_class->finalize = gst_xcam_filter_finalize;
    gobject_class->set_property = gst_xcam_filter_set_property;
    gobject_class->get_property = gst_xcam_filter_get_property;

    g_object_class_install_property (
        gobject_class, PROP_BUFFERCOUNT,
        g_param_spec_int ("buffercount", "buffer count", "Buffer count",
                          0, G_MAXINT, DEFAULT_PROP_BUFFERCOUNT,
                          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property (
        gobject_class, PROP_COPY_MODE,
        g_param_spec_enum ("copy-mode", "copy mode", "Copy Mode",
                           GST_TYPE_XCAM_FILTER_COPY_MODE, DEFAULT_PROP_COPY_MODE,
                           (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property (
        gobject_class, PROP_DEFOG_MODE,
        g_param_spec_enum ("defog-mode", "defog mode", "Defog mode",
                           GST_TYPE_XCAM_FILTER_DEFOG_MODE, DEFAULT_PROP_DEFOG_MODE,
                           (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property (
        gobject_class, PROP_WAVELET_MODE,
        g_param_spec_enum ("wavelet-mode", "wavelet mode", "Wavelet Mode",
                           GST_TYPE_XCAM_FILTER_WAVELET_MODE, DEFAULT_PROP_WAVELET_MODE,
                           (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property (
        gobject_class, PROP_DENOISE_3D_MODE,
        g_param_spec_enum ("denoise-3d", "3D Denoise mode", "3D Denoise mode",
                           GST_TYPE_XCAM_FILTER_3D_DENOISE_MODE, DEFAULT_PROP_3D_DENOISE_MODE,
                           (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property (
        gobject_class, PROP_ENABLE_WIREFRAME,
        g_param_spec_boolean ("enable-wireframe", "enable wire frame", "Enable wire frame",
                              DEFAULT_PROP_ENABLE_WIREFRAME, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property (
        gobject_class, PROP_ENABLE_IMAGE_WARP,
        g_param_spec_boolean ("enable-warp", "enable image warp", "Enable Image Warp",
                              DEFAULT_PROP_ENABLE_IMAGE_WARP, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property (
        gobject_class, PROP_ENABLE_IMAGE_STITCH,
        g_param_spec_boolean ("enable-stitch", "enable image stitch", "Enable Image Stitch",
                              DEFAULT_PROP_ENABLE_IMAGE_STITCH, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property (
        gobject_class, PROP_STITCH_ENABLE_SEAM,
        g_param_spec_boolean ("stitch-seam", "enable seam just for stitch", "Enable Seam Just For Stitch",
                              DEFAULT_PROP_STITCH_ENABLE_SEAM, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property (
        gobject_class, PROP_STITCH_SCALE_MODE,
        g_param_spec_enum ("stitch-scale", "stitch scale mode", "Stitch Scale Mode",
                           GST_TYPE_XCAM_FILTER_STITCH_SCALE_MODE, DEFAULT_PROP_STITCH_SCALE_MODE,
                           (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property (
        gobject_class, PROP_STITCH_FISHEYE_MAP,
        g_param_spec_boolean ("stitch-fisheye-map", "stitch fisheye map", "Enable fisheye map for stitch",
                              DEFAULT_PROP_STITCH_FISHEYE_MAP, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property (
        gobject_class, PROP_STITCH_LSC,
        g_param_spec_boolean ("stitch-lsc", "stitch enable lens shading correction", "Enable Lens Shading Correction",
                              DEFAULT_PROP_STITCH_LSC, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

#if HAVE_OPENCV
    g_object_class_install_property (
        gobject_class, PROP_STITCH_FM_OCL,
        g_param_spec_boolean ("stitch-fm-ocl", "stitch enable ocl for feature match", "Enable ocl for feature match",
                              DEFAULT_PROP_STITCH_FM_OCL, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
#endif

    g_object_class_install_property (
        gobject_class, PROP_STITCH_RES_MODE,
        g_param_spec_enum ("stitch-res-mode", "stitch resolution mode", "Stitch Resolution Mode",
                           GST_TYPE_XCAM_FILTER_STITCH_RES_MODE, DEFAULT_PROP_STITCH_RES_MODE,
                           (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    gst_element_class_set_details_simple (element_class,
                                          "Libxcam Filter",
                                          "Filter/Effect/Video",
                                          "Process NV12 stream using xcam library",
                                          "Wind Yuan <feng.yuan@intel.com> & Yinhang Liu <yinhangx.liu@intel.com>");

    gst_element_class_add_pad_template (element_class,
                                        gst_static_pad_template_get (&gst_xcam_src_factory));
    gst_element_class_add_pad_template (element_class,
                                        gst_static_pad_template_get (&gst_xcam_sink_factory));

    basetrans_class->start = GST_DEBUG_FUNCPTR (gst_xcam_filter_start);
    basetrans_class->stop = GST_DEBUG_FUNCPTR (gst_xcam_filter_stop);
    basetrans_class->transform_caps = GST_DEBUG_FUNCPTR (gst_xcam_filter_transform_caps);
    basetrans_class->set_caps = GST_DEBUG_FUNCPTR (gst_xcam_filter_set_caps);
    basetrans_class->before_transform = GST_DEBUG_FUNCPTR (gst_xcam_filter_before_transform);
    basetrans_class->prepare_output_buffer = GST_DEBUG_FUNCPTR (gst_xcam_filter_prepare_output_buffer);
    basetrans_class->transform = GST_DEBUG_FUNCPTR (gst_xcam_filter_transform);
}

static void
gst_xcam_filter_init (GstXCamFilter *xcamfilter)
{
    xcamfilter->buf_count = DEFAULT_PROP_BUFFERCOUNT;
    xcamfilter->copy_mode = DEFAULT_PROP_COPY_MODE;
    xcamfilter->defog_mode = DEFAULT_PROP_DEFOG_MODE;
    xcamfilter->wavelet_mode = DEFAULT_PROP_WAVELET_MODE;
    xcamfilter->denoise_3d_mode = DEFAULT_PROP_3D_DENOISE_MODE;
    xcamfilter->denoise_3d_ref_count = 2;
    xcamfilter->enable_wireframe = DEFAULT_PROP_ENABLE_WIREFRAME;
    xcamfilter->enable_image_warp = DEFAULT_PROP_ENABLE_IMAGE_WARP;
    xcamfilter->enable_stitch = DEFAULT_PROP_ENABLE_IMAGE_STITCH;
    xcamfilter->stitch_enable_seam = DEFAULT_PROP_STITCH_ENABLE_SEAM;
    xcamfilter->stitch_fisheye_map = DEFAULT_PROP_STITCH_FISHEYE_MAP;
    xcamfilter->stitch_lsc = DEFAULT_PROP_STITCH_LSC;
    xcamfilter->stitch_fm_ocl = DEFAULT_PROP_STITCH_FM_OCL;
    xcamfilter->stitch_scale_mode = DEFAULT_PROP_STITCH_SCALE_MODE;
    xcamfilter->stitch_res_mode = DEFAULT_PROP_STITCH_RES_MODE;

    xcamfilter->delay_buf_num = DEFAULT_DELAY_BUFFER_NUM;
    xcamfilter->cached_buf_num = 0;

    XCAM_CONSTRUCTOR (xcamfilter->pipe_manager, SmartPtr<MainPipeManager>);
    xcamfilter->pipe_manager = new MainPipeManager;
    XCAM_ASSERT (xcamfilter->pipe_manager.ptr ());
}

static void
gst_xcam_filter_finalize (GObject *object)
{
    GstXCamFilter *xcamfilter = GST_XCAM_FILTER (object);

    if (xcamfilter->allocator)
        gst_object_unref (xcamfilter->allocator);

    xcamfilter->pipe_manager.release ();
    XCAM_DESTRUCTOR (xcamfilter->pipe_manager, SmartPtr<MainPipeManager>);

    G_OBJECT_CLASS (parent_class)->finalize (object);
}

static void
gst_xcam_filter_set_property (GObject *object, guint prop_id, const GValue *value, GParamSpec *pspec)
{
    GstXCamFilter *xcamfilter = GST_XCAM_FILTER (object);

    switch (prop_id) {
    case PROP_BUFFERCOUNT:
        xcamfilter->buf_count = g_value_get_int (value);
        break;
    case PROP_COPY_MODE:
        xcamfilter->copy_mode = (CopyMode) g_value_get_enum (value);
        break;
    case PROP_DEFOG_MODE:
        xcamfilter->defog_mode = (DefogModeType) g_value_get_enum (value);
        break;
    case PROP_WAVELET_MODE:
        xcamfilter->wavelet_mode = (WaveletModeType) g_value_get_enum (value);
        break;
    case PROP_DENOISE_3D_MODE:
        xcamfilter->denoise_3d_mode = (Denoise3DModeType) g_value_get_enum (value);
        break;
    case PROP_ENABLE_WIREFRAME:
        xcamfilter->enable_wireframe = g_value_get_boolean (value);
        break;
    case PROP_ENABLE_IMAGE_WARP:
        xcamfilter->enable_image_warp = g_value_get_boolean (value);
        break;
    case PROP_ENABLE_IMAGE_STITCH:
        xcamfilter->enable_stitch = g_value_get_boolean (value);
        break;
    case PROP_STITCH_ENABLE_SEAM:
        xcamfilter->stitch_enable_seam = g_value_get_boolean (value);
        break;
    case PROP_STITCH_SCALE_MODE:
        xcamfilter->stitch_scale_mode = (CLBlenderScaleMode) g_value_get_enum (value);
        break;
    case PROP_STITCH_FISHEYE_MAP:
        xcamfilter->stitch_fisheye_map = g_value_get_boolean (value);
        break;
    case PROP_STITCH_LSC:
        xcamfilter->stitch_lsc = g_value_get_boolean (value);
        break;
#if HAVE_OPENCV
    case PROP_STITCH_FM_OCL:
        xcamfilter->stitch_fm_ocl = g_value_get_boolean (value);
        break;
#endif
    case PROP_STITCH_RES_MODE:
        xcamfilter->stitch_res_mode = (StitchResMode) g_value_get_enum (value);
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
        break;
    }
}

static void
gst_xcam_filter_get_property (GObject *object, guint prop_id, GValue *value, GParamSpec *pspec)
{
    GstXCamFilter *xcamfilter = GST_XCAM_FILTER (object);

    switch (prop_id) {
    case PROP_BUFFERCOUNT:
        g_value_set_int (value, xcamfilter->buf_count);
        break;
    case PROP_COPY_MODE:
        g_value_set_enum (value, xcamfilter->copy_mode);
        break;
    case PROP_DEFOG_MODE:
        g_value_set_enum (value, xcamfilter->defog_mode);
        break;
    case PROP_WAVELET_MODE:
        g_value_set_enum (value, xcamfilter->wavelet_mode);
        break;
    case PROP_DENOISE_3D_MODE:
        g_value_set_enum (value, xcamfilter->denoise_3d_mode);
        break;
    case PROP_ENABLE_WIREFRAME:
        g_value_set_boolean (value, xcamfilter->enable_wireframe);
        break;
    case PROP_ENABLE_IMAGE_WARP:
        g_value_set_boolean (value, xcamfilter->enable_image_warp);
        break;
    case PROP_ENABLE_IMAGE_STITCH:
        g_value_set_boolean (value, xcamfilter->enable_stitch);
        break;
    case PROP_STITCH_ENABLE_SEAM:
        g_value_set_boolean (value, xcamfilter->stitch_enable_seam);
        break;
    case PROP_STITCH_SCALE_MODE:
        g_value_set_enum (value, xcamfilter->stitch_scale_mode);
        break;
    case PROP_STITCH_FISHEYE_MAP:
        g_value_set_boolean (value, xcamfilter->stitch_fisheye_map);
        break;
    case PROP_STITCH_LSC:
        g_value_set_boolean (value, xcamfilter->stitch_lsc);
        break;
#if HAVE_OPENCV
    case PROP_STITCH_FM_OCL:
        g_value_set_boolean (value, xcamfilter->stitch_fm_ocl);
        break;
#endif
    case PROP_STITCH_RES_MODE:
        g_value_set_enum (value, xcamfilter->stitch_res_mode);
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
        break;
    }
}

static gboolean
gst_xcam_filter_start (GstBaseTransform *trans)
{
    GstXCamFilter *xcamfilter = GST_XCAM_FILTER (trans);

    if (xcamfilter->buf_count <= xcamfilter->delay_buf_num) {
        XCAM_LOG_ERROR (
            "buffer count (%d) should be greater than delayed buffer number (%d)",
            xcamfilter->buf_count,
            xcamfilter->delay_buf_num);
        return false;
    }

    SmartPtr<MainPipeManager> pipe_manager = xcamfilter->pipe_manager;
    SmartPtr<SmartAnalyzer> smart_analyzer;
    SmartPtr<CLPostImageProcessor> image_processor;

    SmartHandlerList smart_handlers = SmartAnalyzerLoader::load_smart_handlers (DEFAULT_SMART_ANALYSIS_LIB_DIR);
    if (!smart_handlers.empty ()) {
        smart_analyzer = new SmartAnalyzer ();
        if (smart_analyzer.ptr ()) {
            SmartHandlerList::iterator i_handler = smart_handlers.begin ();
            for (; i_handler != smart_handlers.end ();  ++i_handler)
            {
                XCAM_ASSERT ((*i_handler).ptr ());
                smart_analyzer->add_handler (*i_handler);
            }
            if (smart_analyzer->prepare_handlers () != XCAM_RETURN_NO_ERROR) {
                XCAM_LOG_WARNING ("analyzer(%s) prepare handlers failed", smart_analyzer->get_name ());
                return false;
            }
            pipe_manager->set_smart_analyzer (smart_analyzer);
        } else {
            XCAM_LOG_WARNING ("load smart analyzer(%s) failed, please check.", DEFAULT_SMART_ANALYSIS_LIB_DIR);
        }
    }

    image_processor = new CLPostImageProcessor ();
    XCAM_ASSERT (image_processor.ptr ());
    image_processor->set_stats_callback (pipe_manager);
    image_processor->set_defog_mode ((CLPostImageProcessor::CLDefogMode) xcamfilter->defog_mode);

    if (NONE_WAVELET != xcamfilter->wavelet_mode) {
        if (HAT_WAVELET_Y == xcamfilter->wavelet_mode) {
            image_processor->set_wavelet (CL_WAVELET_HAT, CL_IMAGE_CHANNEL_Y, false);
        } else if (HAT_WAVELET_UV == xcamfilter->wavelet_mode) {
            image_processor->set_wavelet (CL_WAVELET_HAT, CL_IMAGE_CHANNEL_UV, false);
        } else if (HARR_WAVELET_Y == xcamfilter->wavelet_mode) {
            image_processor->set_wavelet (CL_WAVELET_HAAR, CL_IMAGE_CHANNEL_Y, false);
        } else if (HARR_WAVELET_UV == xcamfilter->wavelet_mode) {
            image_processor->set_wavelet (CL_WAVELET_HAAR, CL_IMAGE_CHANNEL_UV, false);
        } else if (HARR_WAVELET_YUV == xcamfilter->wavelet_mode) {
            image_processor->set_wavelet (CL_WAVELET_HAAR, CL_IMAGE_CHANNEL_UV | CL_IMAGE_CHANNEL_Y, false);
        } else if (HARR_WAVELET_BAYES == xcamfilter->wavelet_mode) {
            image_processor->set_wavelet (CL_WAVELET_HAAR, CL_IMAGE_CHANNEL_UV | CL_IMAGE_CHANNEL_Y, true);
        } else {
            image_processor->set_wavelet (CL_WAVELET_DISABLED, CL_IMAGE_CHANNEL_UV, false);
        }
    }

    image_processor->set_3ddenoise_mode (
        (CLPostImageProcessor::CL3DDenoiseMode) xcamfilter->denoise_3d_mode, xcamfilter->denoise_3d_ref_count);

    image_processor->set_wireframe (xcamfilter->enable_wireframe);
    image_processor->set_image_warp (xcamfilter->enable_image_warp);
    if (smart_analyzer.ptr ()) {
        if (xcamfilter->enable_wireframe)
            image_processor->set_scaler (true);

        if (xcamfilter->enable_image_warp) {
            image_processor->set_scaler (true);
            xcamfilter->delay_buf_num = DEFAULT_DELAY_BUFFER_NUM + 16;
        }
    }

    pipe_manager->add_image_processor (image_processor);
    pipe_manager->set_image_processor (image_processor);

    if (xcamfilter->copy_mode == COPY_MODE_DMA) {
        xcamfilter->allocator = gst_dmabuf_allocator_new ();
        if (!xcamfilter->allocator) {
            GST_WARNING ("xcamfilter get allocator failed");
            return false;
        }
    }

    SmartPtr<DrmDisplay> drm_disp = DrmDisplay::instance ();
    xcamfilter->buf_pool = new DrmBoBufferPool (drm_disp);
    XCAM_ASSERT (xcamfilter->buf_pool.ptr ());

    return true;
}

static gboolean
gst_xcam_filter_stop (GstBaseTransform *trans)
{
    GstXCamFilter *xcamfilter = GST_XCAM_FILTER (trans);

    SmartPtr<DrmBoBufferPool> buf_pool = xcamfilter->buf_pool;
    if (buf_pool.ptr ())
        buf_pool->stop ();

    SmartPtr<MainPipeManager> pipe_manager = xcamfilter->pipe_manager;
    if (pipe_manager.ptr ())
        pipe_manager->stop ();

    return true;
}

static GstCaps *
gst_xcam_filter_transform_caps (
    GstBaseTransform *trans, GstPadDirection direction, GstCaps *caps, GstCaps *filter)
{
    GstXCamFilter *xcamfilter = GST_XCAM_FILTER (trans);

    GstCaps *src_caps, *peer_caps, *intersect_caps;
    GstStructure *sink_struct, *src_struct;
    GstPad *peer_pad;
    gint sink_width, sink_height, src_width, src_height;

    gboolean is_sink_width = false;
    gboolean is_sink_height = false;

    src_caps = gst_pad_get_pad_template_caps (trans->srcpad);

    if (direction == GST_PAD_SRC || !gst_caps_is_fixed (caps))
        goto filtering;

    sink_struct = gst_caps_get_structure (caps, 0);
    if (!gst_structure_get_int (sink_struct, "width", &sink_width) ||
            !gst_structure_get_int (sink_struct, "height", &sink_height))
        goto filtering;

    peer_pad = gst_pad_get_peer (trans->srcpad);
    peer_caps = gst_pad_query_caps (peer_pad, src_caps);
    if (!peer_pad || gst_caps_is_empty (peer_caps)) {
        if (xcamfilter->enable_stitch) {
            src_height = XCAM_ALIGN_UP (sink_width / 2, 16);
            if (src_height * 2 != sink_width) {
                gst_caps_unref (src_caps);
                gst_caps_unref (peer_caps);
                XCAM_LOG_ERROR ("xcamfilter stitch incorrect size, sink-width(%d) / 2 should be aligned with 16",
                                sink_width);
                return NULL;
            }
            src_width = sink_width;

            gst_caps_unref (src_caps);
            src_caps = gst_caps_copy (caps);
            src_struct = gst_caps_get_structure (src_caps, 0);

            gst_structure_set (src_struct, "width", G_TYPE_INT, src_width,
                               "height", G_TYPE_INT, src_height, NULL);
        }

        gst_caps_unref (peer_caps);
        goto filtering;
    }

    intersect_caps = gst_caps_intersect_full (peer_caps, src_caps, GST_CAPS_INTERSECT_FIRST);
    gst_caps_unref (src_caps);
    src_caps = intersect_caps;

    src_struct = gst_caps_get_structure (src_caps, 0);
    if (!gst_structure_get_int (src_struct, "width", &src_width)) {
        is_sink_width = true;
        src_width = sink_width;
    }
    if (!gst_structure_get_int (src_struct, "height", &src_height)) {
        is_sink_height = true;
        src_height = sink_height;
    }

    if (xcamfilter->enable_stitch) {
        if (is_sink_width && is_sink_height)
            src_height = XCAM_ALIGN_UP (src_width / 2, 16);

        if (src_width != src_height * 2) {
            XCAM_LOG_ERROR ("xcamfilter incorrect stitch size width:%d height:%d", src_width, src_height);
            gst_caps_unref (src_caps);
            return NULL;
        }
    }

    gint fps_n, fps_d;
    if (!gst_structure_get_fraction (src_struct, "framerate", &fps_n, &fps_d) &&
            !gst_structure_get_fraction (sink_struct, "framerate", &fps_n, &fps_d)) {
        fps_n = 25;
        fps_d = 1;
    }

    gst_structure_set (src_struct, "width", G_TYPE_INT, src_width,
                       "height", G_TYPE_INT, src_height,
                       "framerate", GST_TYPE_FRACTION, fps_n, fps_d, NULL);

filtering:
    if (filter) {
        intersect_caps = gst_caps_intersect_full (filter, src_caps, GST_CAPS_INTERSECT_FIRST);
        gst_caps_unref (src_caps);
        src_caps = intersect_caps;
    }

    return src_caps;
}

static gboolean
gst_xcam_filter_set_caps (GstBaseTransform *trans, GstCaps *incaps, GstCaps *outcaps)
{
    GstXCamFilter *xcamfilter = GST_XCAM_FILTER (trans);
    GstVideoInfo in_info, out_info;

    if (!gst_video_info_from_caps (&in_info, incaps) ||
            !gst_video_info_from_caps (&out_info, outcaps)) {
        XCAM_LOG_WARNING ("fail to parse incaps or outcaps");
        return false;
    }

    XCAM_FAIL_RETURN (
        ERROR,
        GST_VIDEO_INFO_FORMAT (&in_info) == GST_VIDEO_FORMAT_NV12 ||
        GST_VIDEO_INFO_FORMAT (&out_info) == GST_VIDEO_FORMAT_NV12,
        false,
        "xcamfilter only support NV12 stream");
    xcamfilter->gst_sink_video_info = in_info;
    xcamfilter->gst_src_video_info = out_info;

    SmartPtr<MainPipeManager> pipe_manager = xcamfilter->pipe_manager;
    SmartPtr<CLPostImageProcessor> processor = pipe_manager->get_image_processor();
    XCAM_ASSERT (pipe_manager.ptr () && processor.ptr ());
    if (!processor->set_output_format (V4L2_PIX_FMT_NV12))
        return false;

    if (processor->is_scaled ())
        processor->set_scaler_factor (640.0 / GST_VIDEO_INFO_WIDTH (&in_info));
    //processor->set_scaler_factor (0.5f);

    if (xcamfilter->enable_stitch) {
        processor->set_image_stitch (
            xcamfilter->enable_stitch, xcamfilter->stitch_enable_seam, xcamfilter->stitch_scale_mode,
            xcamfilter->stitch_fisheye_map, xcamfilter->stitch_lsc, xcamfilter->stitch_fm_ocl,
            GST_VIDEO_INFO_WIDTH (&out_info), GST_VIDEO_INFO_HEIGHT (&out_info), (uint32_t) xcamfilter->stitch_res_mode);
        XCAM_LOG_INFO ("xcamfilter stitch output size width:%d height:%d",
                       GST_VIDEO_INFO_WIDTH (&out_info), GST_VIDEO_INFO_HEIGHT (&out_info));
    }

    if (pipe_manager->start () != XCAM_RETURN_NO_ERROR) {
        XCAM_LOG_ERROR ("pipe manager start failed");
        return false;
    }

    VideoBufferInfo buf_info;
    buf_info.init (
        V4L2_PIX_FMT_NV12,
        GST_VIDEO_INFO_WIDTH (&in_info),
        GST_VIDEO_INFO_HEIGHT (&in_info),
        XCAM_ALIGN_UP (GST_VIDEO_INFO_WIDTH (&in_info), 16),
        XCAM_ALIGN_UP (GST_VIDEO_INFO_HEIGHT (&in_info), 16));

    SmartPtr<DrmBoBufferPool> buf_pool = xcamfilter->buf_pool;
    XCAM_ASSERT (buf_pool.ptr ());
    if (!buf_pool->set_video_info (buf_info) ||
            !buf_pool->reserve (xcamfilter->buf_count)) {
        XCAM_LOG_ERROR ("init buffer pool failed");
        return false;
    }

    return true;
}

static GstFlowReturn
copy_gstbuf_to_xcambuf (GstVideoInfo gstinfo, GstBuffer *gstbuf, SmartPtr<VideoBuffer> xcambuf)
{
    GstMapInfo mapinfo;
    VideoBufferPlanarInfo planar;
    const VideoBufferInfo xcaminfo = xcambuf->get_video_info ();

    uint8_t *memory = xcambuf->map ();
    gboolean ret = gst_buffer_map (gstbuf, &mapinfo, GST_MAP_READ);
    if (!memory || !ret) {
        XCAM_LOG_WARNING ("xcamfilter map buffer failed");
        return GST_FLOW_ERROR;
    }

    uint8_t *src = NULL;
    uint8_t *dest = NULL;
    for (uint32_t index = 0; index < xcaminfo.components; index++) {
        xcaminfo.get_planar_info (planar, index);

        src = mapinfo.data + GST_VIDEO_INFO_PLANE_OFFSET (&gstinfo, index);
        dest = memory + xcaminfo.offsets [index];
        for (uint32_t i = 0; i < planar.height; i++) {
            memcpy (dest, src, GST_VIDEO_INFO_WIDTH (&gstinfo));
            src += GST_VIDEO_INFO_PLANE_STRIDE (&gstinfo, index);
            dest += xcaminfo.strides [index];
        }
    }

    gst_buffer_unmap (gstbuf, &mapinfo);
    xcambuf->unmap ();

    return GST_FLOW_OK;
}

static GstFlowReturn
copy_xcambuf_to_gstbuf (GstVideoInfo gstinfo, SmartPtr<VideoBuffer> xcambuf, GstBuffer **gstbuf)
{
    GstMapInfo mapinfo;
    VideoBufferPlanarInfo planar;
    const VideoBufferInfo xcaminfo = xcambuf->get_video_info ();

    GstBuffer *tmpbuf = gst_buffer_new_allocate (NULL, GST_VIDEO_INFO_SIZE (&gstinfo), NULL);
    if (!tmpbuf) {
        XCAM_LOG_ERROR ("xcamfilter allocate buffer failed");
        return GST_FLOW_ERROR;
    }

    uint8_t *memory = xcambuf->map ();
    gboolean ret = gst_buffer_map (tmpbuf, &mapinfo, GST_MAP_WRITE);
    if (!memory || !ret) {
        XCAM_LOG_WARNING ("xcamfilter map buffer failed");
        return GST_FLOW_ERROR;
    }

    uint8_t *src = NULL;
    uint8_t *dest = NULL;
    for (uint32_t index = 0; index < GST_VIDEO_INFO_N_PLANES (&gstinfo); index++) {
        xcaminfo.get_planar_info (planar, index);

        src = memory + xcaminfo.offsets [index];
        dest = mapinfo.data + GST_VIDEO_INFO_PLANE_OFFSET (&gstinfo, index);
        for (uint32_t i = 0; i < planar.height; i++) {
            memcpy (dest, src, planar.width);
            src += xcaminfo.strides [index];
            dest += GST_VIDEO_INFO_PLANE_STRIDE (&gstinfo, index);
        }
    }

    gst_buffer_unmap (tmpbuf, &mapinfo);
    xcambuf->unmap ();

    *gstbuf = tmpbuf;

    return GST_FLOW_OK;
}

static GstFlowReturn
append_xcambuf_to_gstbuf (GstAllocator *allocator, SmartPtr<VideoBuffer> xcambuf, GstBuffer **gstbuf)
{
    gsize offsets [XCAM_VIDEO_MAX_COMPONENTS];

    VideoBufferInfo xcaminfo = xcambuf->get_video_info ();
    for (int i = 0; i < XCAM_VIDEO_MAX_COMPONENTS; i++) {
        offsets [i] = xcaminfo.offsets [i];
    }

    GstBuffer *tmpbuf = gst_buffer_new ();
    GstMemory *mem = gst_dmabuf_allocator_alloc (allocator, dup (xcambuf->get_fd ()), xcambuf->get_size ());
    XCAM_ASSERT (mem);

    gst_buffer_append_memory (tmpbuf, mem);

    gst_buffer_add_video_meta_full (
        tmpbuf,
        GST_VIDEO_FRAME_FLAG_NONE,
        GST_VIDEO_FORMAT_NV12,
        xcaminfo.width,
        xcaminfo.height,
        xcaminfo.components,
        offsets,
        (gint *) (xcaminfo.strides));

    *gstbuf = tmpbuf;

    return GST_FLOW_OK;
}

static gint
get_dmabuf_fd (GstBuffer *buffer)
{
    GstMemory *mem = gst_buffer_peek_memory (buffer, 0);
    if (!gst_is_dmabuf_memory (mem)) {
        return -1;
    }

    return gst_dmabuf_memory_get_fd (mem);
}

static void
gst_xcam_filter_before_transform (GstBaseTransform *trans, GstBuffer *buffer)
{
    GstXCamFilter *xcamfilter = GST_XCAM_FILTER (trans);

    SmartPtr<DrmBoBufferPool> buf_pool = xcamfilter->buf_pool;
    SmartPtr<MainPipeManager> pipe_manager = xcamfilter->pipe_manager;
    XCAM_ASSERT (buf_pool.ptr () && pipe_manager.ptr ());

    if (xcamfilter->cached_buf_num > xcamfilter->delay_buf_num)
        return;

    SmartPtr<VideoBuffer> video_buf;
    gint dma_fd = get_dmabuf_fd (buffer);
    if (dma_fd >= 0) {
        SmartPtr<DrmDisplay> display = buf_pool->get_drm_display ();
        VideoBufferInfo info = buf_pool->get_video_info ();

        SmartPtr<VideoBuffer> dma_buf = new DmaGstBuffer (info, dma_fd, buffer);
        video_buf = display->convert_to_drm_bo_buf (display, dma_buf);
        if (!video_buf.ptr ()) {
            XCAM_LOG_ERROR ("xcamfilter convert to drm bo buffer failed");
            return;
        }
    } else {
        SmartPtr<DrmBoBuffer> drm_buf = buf_pool->get_buffer (buf_pool).dynamic_cast_ptr<DrmBoBuffer> ();
        if (!drm_buf.ptr ()) {
            XCAM_LOG_ERROR ("xcamfilter sink-pad get drm buffer failed");
            return;
        }

        video_buf = drm_buf;
        copy_gstbuf_to_xcambuf (xcamfilter->gst_sink_video_info, buffer, video_buf);
    }

    if (pipe_manager->push_buffer (video_buf) != XCAM_RETURN_NO_ERROR) {
        XCAM_LOG_ERROR ("xcamfilter push buffer failed");
        return;
    }

    xcamfilter->cached_buf_num++;
}

static GstFlowReturn
gst_xcam_filter_prepare_output_buffer (GstBaseTransform *trans, GstBuffer *input, GstBuffer **outbuf)
{
    GstXCamFilter *xcamfilter = GST_XCAM_FILTER (trans);
    GstFlowReturn ret = GST_FLOW_OK;

    SmartPtr<MainPipeManager> pipe_manager = xcamfilter->pipe_manager;
    SmartPtr<VideoBuffer> video_buf;

    if (xcamfilter->cached_buf_num > xcamfilter->buf_count)
        return GST_FLOW_ERROR;

    int32_t timeout = -1;
    if (xcamfilter->cached_buf_num <= xcamfilter->delay_buf_num)
        timeout = 0;

    video_buf = pipe_manager->dequeue_buffer (timeout);
    if (!video_buf.ptr ()) {
        XCAM_LOG_WARNING ("xcamfilter dequeue buffer failed");
        *outbuf = NULL;
        return GST_FLOW_OK;
    }

    if (xcamfilter->copy_mode == COPY_MODE_CPU) {
        ret = copy_xcambuf_to_gstbuf (xcamfilter->gst_src_video_info, video_buf, outbuf);
    } else if (xcamfilter->copy_mode == COPY_MODE_DMA) {
        GstAllocator *allocator = xcamfilter->allocator;
        ret = append_xcambuf_to_gstbuf (allocator, video_buf, outbuf);
    }

    if (ret == GST_FLOW_OK) {
        xcamfilter->cached_buf_num--;
        GST_BUFFER_TIMESTAMP (*outbuf) = GST_BUFFER_TIMESTAMP (input);
    }

    return ret;
}

static GstFlowReturn
gst_xcam_filter_transform (GstBaseTransform *trans, GstBuffer *inbuf, GstBuffer *outbuf)
{
    XCAM_UNUSED (trans);
    XCAM_UNUSED (inbuf);

    if (!outbuf) {
        XCAM_LOG_ERROR ("transform failed with null outbufer");
        return GST_FLOW_ERROR;
    }

    XCAM_STATIC_FPS_CALCULATION (gstxcamfilter, XCAM_OBJ_DUR_FRAME_NUM);
    return GST_FLOW_OK;
}

static gboolean
gst_xcam_filter_plugin_init (GstPlugin *xcamfilter)
{
    return gst_element_register (xcamfilter, "xcamfilter", GST_RANK_NONE,
                                 GST_TYPE_XCAM_FILTER);
}

#ifndef PACKAGE
#define PACKAGE "libxam"
#endif

GST_PLUGIN_DEFINE (
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    xcamfilter,
    "Libxcam filter plugin",
    gst_xcam_filter_plugin_init,
    VERSION,
    GST_LICENSE_UNKNOWN,
    "libxcamfilter",
    "https://github.com/01org/libxcam"
)
