/*
 * gstxcambufferpool.c - bufferpool
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
 * SECTION:element-xcambufferpool
 *
 * FIXME:Describe xcambufferpool here.
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch -v -m fakesrc ! xcambufferpool ! fakesink silent=TRUE
 * ]|
 * </refsect2>
 */

#ifdef HAVE_CONFIG_H
#  include <config.h>
#endif

#include <stdint.h>

#include <gst/gst.h>
#include <gst/video/gstvideopool.h>

#include "gstxcambufferpool.h"

GST_DEBUG_CATEGORY_STATIC (gst_xcam_debug);
#define GST_CAT_DEFAULT gst_xcam_debug

static gboolean
gst_xcambufferpool_start (GstBufferPool *bpool);

static GstFlowReturn
gst_xcambufferpool_acquire_buffer (GstBufferPool *bpool, GstBuffer **buffer, GstBufferPoolAcquireParams *params);

static void
gst_xcambufferpool_release_buffer (GstBufferPool *bpool, GstBuffer *buffer);

#define gst_xcambufferpool_parent_class parent_class
G_DEFINE_TYPE (Gstxcambufferpool, gst_xcambufferpool, GST_TYPE_BUFFER_POOL);

static void
gst_xcambufferpool_class_init (GstxcambufferpoolClass * klass)
{
    GObjectClass *object_class;
    GstBufferPoolClass *bufferpool_class;

    object_class = G_OBJECT_CLASS (klass);
    bufferpool_class = GST_BUFFER_POOL_CLASS (klass);

    bufferpool_class->start = gst_xcambufferpool_start;
    bufferpool_class->acquire_buffer = gst_xcambufferpool_acquire_buffer;
    bufferpool_class->release_buffer = gst_xcambufferpool_release_buffer;

}

static void
gst_xcambufferpool_init (Gstxcambufferpool *pool)
{
}

static gboolean
gst_xcambufferpool_start (GstBufferPool *bpool)
{
    Gstxcambufferpool *pool = GST_XCAMBUFFERPOOL_CAST (bpool);

    libxcam_start ();

    pool->allocator = gst_dmabuf_allocator_new();
    if (pool->allocator == NULL) {
        printf ("gst_xcambufferpool_new::gst_dmabuf_allocator_new failed\n");
        return NULL;
    }

    return TRUE;
}

static GstFlowReturn
gst_xcambufferpool_acquire_buffer (GstBufferPool *bpool, GstBuffer **buffer, GstBufferPoolAcquireParams *params)
{
    return xcam_bufferpool_acquire_buffer (bpool, buffer, params);
}

static void
gst_xcambufferpool_release_buffer (GstBufferPool *bpool, GstBuffer *buffer)
{
    return xcambufferpool_release_buffer (bpool, buffer);
}


GstBufferPool *
gst_xcambufferpool_new (Gstxcamsrc *xcamsrc, GstCaps *caps)
{
    Gstxcambufferpool *pool;
    GstStructure *s;

    pool = (Gstxcambufferpool *)g_object_new (GST_TYPE_XCAMBUFFERPOOL, NULL);
    s = gst_buffer_pool_get_config (GST_BUFFER_POOL_CAST (pool));
    gst_buffer_pool_config_set_params (s, caps, 1999, 32, 32);
    gst_buffer_pool_config_add_option (s, GST_BUFFER_POOL_OPTION_VIDEO_META);
    gst_buffer_pool_set_config (GST_BUFFER_POOL_CAST (pool), s);

    pool->src = xcamsrc;
    return GST_BUFFER_POOL (pool);
}
