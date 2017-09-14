/*
 * gstxcambufferpool.h - buffer pool
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

#ifndef GST_XCAM_BUFFER_POOL_H
#define GST_XCAM_BUFFER_POOL_H

#include <gst/gst.h>
#include "main_dev_manager.h"
#include "gstxcamsrc.h"

G_BEGIN_DECLS

#define GST_TYPE_XCAM_BUFFER_POOL \
  (gst_xcam_buffer_pool_get_type())
#define GST_XCAM_BUFFER_POOL(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_XCAM_BUFFER_POOL,GstXCamBufferPool))

typedef struct _GstXCamBufferPool      GstXCamBufferPool;
typedef struct _GstXCamBufferPoolClass GstXCamBufferPoolClass;

struct _GstXCamBufferPool
{
    GstBufferPool                              parent;
    GstAllocator                              *allocator;
    GstXCamSrc                                *src;
    gboolean                                   need_video_meta;
    XCam::SmartPtr<GstXCam::MainDeviceManager> device_manager;
};

struct _GstXCamBufferPoolClass
{
    GstBufferPoolClass parent_class;
};

GType gst_xcam_buffer_pool_get_type (void);

GstBufferPool *
gst_xcam_buffer_pool_new (GstXCamSrc *xcamsrc, GstCaps *caps, XCam::SmartPtr<GstXCam::MainDeviceManager> &device_manager);

G_END_DECLS

#endif // GST_XCAM_BUFFER_POOL_H
