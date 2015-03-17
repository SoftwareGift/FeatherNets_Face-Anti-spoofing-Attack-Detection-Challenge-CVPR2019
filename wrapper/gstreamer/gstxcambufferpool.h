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
 */

#ifndef __GST_XCAMBUFFERPOOL_H__
#define __GST_XCAMBUFFERPOOL_H__

#include <gst/gst.h>
#include "gstxcamsrc.h"

G_BEGIN_DECLS

#define GST_TYPE_XCAMBUFFERPOOL \
  (gst_xcambufferpool_get_type())
#define GST_XCAMBUFFERPOOL(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_XCAMBUFFERPOOL,Gstxcambufferpool))
#define GST_XCAMBUFFERPOOL_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_XCAMBUFFERPOOL,GstxcambufferpoolClass))
#define GST_IS_XCAMBUFFERPOOL(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_XCAMBUFFERPOOL))
#define GST_IS_XCAMBUFFERPOOL_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_XCAMBUFFERPOOL))
#define GST_XCAMBUFFERPOOL_CAST(obj)            ((Gstxcambufferpool *)(obj))
#define GST_XCAMBUFFERPOOL_GET_CLASS (obj)  \
    (G_TYPE_INSTANCE_GET_CLASS ((obj), GST_TYPE_XCAMBUFFERPOOL, GstxcambufferpoolClass))

typedef struct _Gstxcambufferpool      Gstxcambufferpool;
typedef struct _GstxcambufferpoolClass GstxcambufferpoolClass;
typedef struct _GstxcamMeta        GstxcamMeta;

struct _Gstxcambufferpool
{
    GstBufferPool parent;
    GstAllocator *allocator;
    Gstxcamsrc *src;
};

struct _GstxcambufferpoolClass
{
    GstBufferPoolClass parent_class;
};

GType gst_xcambufferpool_get_type (void);

G_END_DECLS

#endif /* __GST_XCAMBUFFERPOOL_H__ */
