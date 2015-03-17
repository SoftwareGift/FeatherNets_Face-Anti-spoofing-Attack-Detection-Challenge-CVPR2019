/*
 * fmt.cpp - deal with the supported formats
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

#include "fmt.h"
#include <linux/videodev2.h>

#include <map>

std::map<uint32_t, GstVideoFormat> fourcc2fmt = {

    { V4L2_PIX_FMT_NV12, GST_VIDEO_FORMAT_NV12 },
    { V4L2_PIX_FMT_NV21, GST_VIDEO_FORMAT_NV21 },
    { V4L2_PIX_FMT_YVU410,  GST_VIDEO_FORMAT_YVU9 },
    { V4L2_PIX_FMT_YUV410,  GST_VIDEO_FORMAT_YUV9 },
    { V4L2_PIX_FMT_YUV420,  GST_VIDEO_FORMAT_I420 },
    { V4L2_PIX_FMT_YUYV,    GST_VIDEO_FORMAT_YUY2 },
    { V4L2_PIX_FMT_YVU420,  GST_VIDEO_FORMAT_YV12 },
    { V4L2_PIX_FMT_UYVY,    GST_VIDEO_FORMAT_UYVY },
    { V4L2_PIX_FMT_YUV411P, GST_VIDEO_FORMAT_Y41B },
    { V4L2_PIX_FMT_YUV422P, GST_VIDEO_FORMAT_Y42B }
#ifdef V4L2_PIX_FMT_YVYU
    , { V4L2_PIX_FMT_YVYU, GST_VIDEO_FORMAT_YVYU}
#endif

};

#define GST_V4L2_MAX_SIZE (1<<15)

void caps_append(GstCaps *caps)
{
    GstStructure *structure;
    std::map<uint32_t, GstVideoFormat>::iterator iter;

    for (iter = fourcc2fmt.begin(); iter != fourcc2fmt.end(); iter++) {
        structure = gst_structure_new ("video/x-raw",
                                       "format", G_TYPE_STRING,
                                       gst_video_format_to_string (iter->second),
                                       NULL);
        gst_structure_set (structure,
                           "width", GST_TYPE_INT_RANGE, 1, GST_V4L2_MAX_SIZE,
                           "height", GST_TYPE_INT_RANGE, 1, GST_V4L2_MAX_SIZE,
                           "framerate", GST_TYPE_FRACTION_RANGE, 0, 1, 100, 1, NULL);
        gst_caps_append_structure (caps, structure);
    }
}
