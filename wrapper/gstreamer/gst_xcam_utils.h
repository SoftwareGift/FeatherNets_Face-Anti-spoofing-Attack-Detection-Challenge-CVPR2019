/*
 * gst_xcam_utils.h - gst xcam utilities
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

#ifndef GST_XCAM_UTILS_H
#define GST_XCAM_UTILS_H

#include "dma_video_buffer.h"

using namespace XCam;

class DmaGstBuffer
    : public DmaVideoBuffer
{
public:
    DmaGstBuffer (const VideoBufferInfo &info, int dma_fd, GstBuffer *gst_buf)
        : DmaVideoBuffer (info, dma_fd)
        , _gst_buf (gst_buf)
    {
        gst_buffer_ref (_gst_buf);
    }

    ~DmaGstBuffer () {
        gst_buffer_unref (_gst_buf);
    }

private:
    XCAM_DEAD_COPY (DmaGstBuffer);

private:
    GstBuffer *_gst_buf;
};

#endif // GST_XCAM_UTILS_H
