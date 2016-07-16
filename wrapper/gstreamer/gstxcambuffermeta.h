/*
 * gstxcambuffermeta.h - gst xcam buffer meta data
 *
 *  Copyright (c) 2014-2015 Intel Corporation
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
 * Author: Wind Yuan <feng.yuan@intel.com>
 */

#ifndef GST_XCAM_BUFFER_META_H
#define GST_XCAM_BUFFER_META_H

#include <gst/gst.h>
#include <gst/gstmeta.h>

#include <video_buffer.h>

using namespace XCam;

XCAM_BEGIN_DECLARE

#define GST_XCAM_META_TAG_XCAM  "xcam"
#define GST_XCAM_META_TAG_BUF   "buf"

#define GST_XCAM_BUFFER_META_API_TYPE  \
    (gst_xcam_buffer_meta_api_get_type ())

#define gst_buffer_get_xcam_buffer_meta(b) \
    ((GstXCamBufferMeta*)gst_buffer_get_meta ((b), GST_XCAM_BUFFER_META_API_TYPE))

typedef struct _GstXCamBufferMeta {
    GstMeta                     meta_base;
    SmartPtr<VideoBuffer>       buffer;
} GstXCamBufferMeta;

GType
gst_xcam_buffer_meta_api_get_type (void);

GstXCamBufferMeta *
gst_buffer_add_xcam_buffer_meta (
    GstBuffer *buffer,
    const SmartPtr<VideoBuffer>  &data);

XCAM_END_DECLARE

#endif //GST_XCAM_BUFFER_META_H
