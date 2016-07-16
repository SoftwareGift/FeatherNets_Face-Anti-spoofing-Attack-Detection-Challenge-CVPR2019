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

using namespace XCam;
using namespace GstXCam;

XCAM_BEGIN_DECLARE

#define GST_TYPE_XCAM_FILTER             (gst_xcam_filter_get_type())
#define GST_XCAM_FILTER(obj)             (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_XCAM_FILTER,GstXCamFilter))
#define GST_XCAM_FILTER_CLASS(klass)     (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_XCAM_FILTER,GstXCamFilterClass))
#define GST_IS_XCAM_FILTER(obj)          (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_XCAM_FILTER))
#define GST_IS_XCAM_FILTER_CLASS(klass)  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_XCAM_FILTER))
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

typedef struct _GstXCamFilter      GstXCamFilter;
typedef struct _GstXCamFilterClass GstXCamFilterClass;

struct _GstXCamFilter
{
    GstBaseTransform             transform;

    uint32_t                     buf_count;
    CopyMode                     copy_mode;
    DefogModeType                defog_mode;

    GstAllocator                 *allocator;
    GstVideoInfo                 gst_video_info;
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
