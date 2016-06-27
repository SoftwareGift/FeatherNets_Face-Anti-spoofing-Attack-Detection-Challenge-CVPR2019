/*
 * gstxcamsrc.h - gst xcamsrc plugin
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
 * Author: Wind Yuan <feng.yuan@intel.com>
 */

#ifndef GST_XCAM_SRC_H
#define GST_XCAM_SRC_H

#include "main_dev_manager.h"
#include <gst/base/gstpushsrc.h>

using namespace XCam;
using namespace GstXCam;

XCAM_BEGIN_DECLARE

#define GST_TYPE_XCAM_SRC \
  (gst_xcam_src_get_type ())
#define GST_XCAM_SRC(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_XCAM_SRC,GstXCamSrc))
#define GST_XCAM_SRC_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_XCAM_SRC,GstXCamSrcClass))
#define GST_IS_XCAM_SRC(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_XCAM_SRC))
#define GST_IS_XCAM_SRC_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_XCAM_SRC))
#define GST_XCAM_SRC_CAST(obj)   ((GstXCamSrc *) obj)

#define GST_XCAM_SRC_MEM_MODE(src) ((GST_XCAM_SRC_CAST(src))->mem_type)
#define GST_XCAM_SRC_FRMAE_DURATION(src) ((GST_XCAM_SRC_CAST(src))->duration)
#define GST_XCAM_SRC_BUF_COUNT(src) ((GST_XCAM_SRC_CAST(src))->buf_count)
#define GST_XCAM_SRC_OUT_VIDEO_INFO(src) (&(GST_XCAM_SRC_CAST(src))->gst_video_info)


typedef enum {
    ISP_IMAGE_PROCESSOR = 0,
    CL_IMAGE_PROCESSOR,
} ImageProcessorType;

typedef enum {
    NONE_WDR = 0,
    GAUSSIAN_WDR,
    HALEQ_WDR,
} WDRModeType;

typedef enum {
    NONE_WAVELET = 0,
    HAT_WAVELET_Y,
    HAT_WAVELET_UV,
    HARR_WAVELET_Y,
    HARR_WAVELET_UV,
    HARR_WAVELET_YUV,
    HARR_WAVELET_BAYES,
} WaveletModeType;

typedef enum {
    SIMPLE_ANALYZER = 0,
    AIQ_TUNER_ANALYZER,
    DYNAMIC_ANALYZER,
    HYBRID_ANALYZER,
} AnalyzerType;

typedef struct _GstXCamSrc      GstXCamSrc;
typedef struct _GstXCamSrcClass GstXCamSrcClass;

struct _GstXCamSrc
{
    GstPushSrc                   pushsrc;
    GstBufferPool               *pool;

    uint32_t                     buf_count;
    uint32_t                     sensor_id;
    uint32_t                     capture_mode;
    char                        *device;
    char                        *path_to_cpf;
    char                        *path_to_3alib;
    gboolean                     enable_3a;
    gboolean                     enable_usb;
    gboolean                     enable_retinex;
    gboolean                     enable_wireframe;
    char                        *path_to_fake;

    gboolean                     time_offset_ready;
    int64_t                      time_offset;
    int64_t                      buf_mark;
    GstClockTime                 duration;

    enum v4l2_memory             mem_type;
    enum v4l2_field              field;
    uint32_t                     in_format;
    uint32_t                     out_format;
    GstVideoInfo                 gst_video_info;
    VideoBufferInfo              xcam_video_info;
    ImageProcessorType           image_processor_type;
    WDRModeType                  wdr_mode_type;
    AnalyzerType                 analyzer_type;
    int32_t                      cl_pipe_profile;
    SmartPtr<MainDeviceManager>  device_manager;
    WaveletModeType              wavelet_mode;
};

struct _GstXCamSrcClass
{
    GstPushSrcClass parent_class;
};

GType gst_xcam_src_get_type (void);

XCAM_END_DECLARE

#endif // GST_XCAM_SRC_H
