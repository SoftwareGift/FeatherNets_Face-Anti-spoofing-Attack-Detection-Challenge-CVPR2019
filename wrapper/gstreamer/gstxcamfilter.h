/*
 * gstxcamfilter.h -gst xcamfilter plugin
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

#ifndef GST_XCAM_FILTER_H
#define GST_XCAM_FILTER_H

#include <gst/gst.h>
#include <gst/video/video.h>

#include "main_pipe_manager.h"
#include "gst_xcam_utils.h"

using namespace XCam;
using namespace GstXCam;

XCAM_BEGIN_DECLARE

#define GST_TYPE_XCAM_FILTER             (gst_xcam_filter_get_type())
#define GST_XCAM_FILTER(obj)             (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_XCAM_FILTER,GstXCamFilter))
#define GST_XCAM_FILTER_CAST(obj)        ((GstXCamFilter *) obj)


typedef enum {
    COPY_MODE_CPU = 0,
    COPY_MODE_DMA
} CopyMode;

typedef enum {
    DEFOG_NONE = 0,
    DEFOG_RETINEX,
    DEFOG_DCP
} DefogModeType;

typedef enum {
    NONE_WAVELET = 0,
    HAT_WAVELET_Y,
    HAT_WAVELET_UV,
    HARR_WAVELET_Y,
    HARR_WAVELET_UV,
    HARR_WAVELET_YUV,
    HARR_WAVELET_BAYES
} WaveletModeType;

typedef enum {
    DENOISE_3D_NONE = 0,
    DENOISE_3D_YUV,
    DENOISE_3D_UV
} Denoise3DModeType;

enum StitchResMode {
    StitchRes1080P = 0,
    StitchRes4K
};

typedef struct _GstXCamFilter      GstXCamFilter;
typedef struct _GstXCamFilterClass GstXCamFilterClass;

struct _GstXCamFilter
{
    GstBaseTransform             transform;

    uint32_t                     buf_count;
    CopyMode                     copy_mode;
    DefogModeType                defog_mode;
    WaveletModeType              wavelet_mode;
    Denoise3DModeType            denoise_3d_mode;
    uint8_t                      denoise_3d_ref_count;
    gboolean                     enable_wireframe;
    gboolean                     enable_image_warp;
    gboolean                     enable_stitch;
    gboolean                     stitch_enable_seam;
    gboolean                     stitch_fisheye_map;
    gboolean                     stitch_fm_ocl;
    gboolean                     stitch_lsc;
    CLBlenderScaleMode           stitch_scale_mode;
    StitchResMode                stitch_res_mode;

    uint32_t                     delay_buf_num;
    uint32_t                     cached_buf_num;
    GstAllocator                 *allocator;
    GstVideoInfo                 gst_sink_video_info;
    GstVideoInfo                 gst_src_video_info;
    SmartPtr<DrmBoBufferPool>    buf_pool;
    SmartPtr<MainPipeManager>    pipe_manager;
};

struct _GstXCamFilterClass
{
    GstBaseTransformClass parent_class;
};

GType gst_xcam_filter_get_type (void);

XCAM_END_DECLARE

#endif // GST_XCAM_FILTER_H
