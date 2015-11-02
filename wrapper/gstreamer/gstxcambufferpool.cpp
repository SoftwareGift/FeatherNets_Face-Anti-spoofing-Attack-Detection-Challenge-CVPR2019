/*
 * gstxcambufferpool.cpp - bufferpool
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

#include "gstxcambufferpool.h"
#include "gstxcambuffermeta.h"

#include <gst/video/gstvideopool.h>
#include <gst/allocators/gstdmabuf.h>
#include <gst/gstmeta.h>

using namespace XCam;
using namespace GstXCam;

XCAM_BEGIN_DECLARE

GST_DEBUG_CATEGORY_EXTERN (gst_xcam_src_debug);
#define GST_CAT_DEFAULT gst_xcam_src_debug

G_DEFINE_TYPE (GstXCamBufferPool, gst_xcam_buffer_pool, GST_TYPE_BUFFER_POOL);
#define parent_class gst_xcam_buffer_pool_parent_class

static void
gst_xcam_buffer_pool_finalize (GObject * object);

static gboolean
gst_xcam_buffer_pool_start (GstBufferPool *pool);

static gboolean
gst_xcam_buffer_pool_stop (GstBufferPool *pool);

static gboolean
gst_xcam_buffer_pool_set_config (GstBufferPool *pool, GstStructure *config);

static GstFlowReturn
gst_xcam_buffer_pool_acquire_buffer (
    GstBufferPool *bpool,
    GstBuffer **buffer,
    GstBufferPoolAcquireParams *params);

static void
gst_xcam_buffer_pool_release_buffer (GstBufferPool *bpool, GstBuffer *buffer);


XCAM_END_DECLARE

static void
gst_xcam_buffer_pool_class_init (GstXCamBufferPoolClass * klass)
{
    GObjectClass *object_class;
    GstBufferPoolClass *bufferpool_class;

    object_class = G_OBJECT_CLASS (klass);
    bufferpool_class = GST_BUFFER_POOL_CLASS (klass);

    object_class->finalize = gst_xcam_buffer_pool_finalize;

    bufferpool_class->start = gst_xcam_buffer_pool_start;
    bufferpool_class->stop = gst_xcam_buffer_pool_stop;
    bufferpool_class->set_config = gst_xcam_buffer_pool_set_config;
    bufferpool_class->acquire_buffer = gst_xcam_buffer_pool_acquire_buffer;
    bufferpool_class->release_buffer = gst_xcam_buffer_pool_release_buffer;

}

static void
gst_xcam_buffer_pool_init (GstXCamBufferPool *pool)
{
    pool->need_video_meta = FALSE;
    XCAM_CONSTRUCTOR (pool->device_manager, SmartPtr<MainDeviceManager>);
}

static void
gst_xcam_buffer_pool_finalize (GObject * object)
{
    GstXCamBufferPool *pool = GST_XCAM_BUFFER_POOL (object);
    XCAM_ASSERT (pool);

    if (pool->src)
        gst_object_unref (pool->src);
    if (pool->allocator)
        gst_object_unref (pool->allocator);
    XCAM_DESTRUCTOR (pool->device_manager, SmartPtr<MainDeviceManager>);
}

static gboolean
gst_xcam_buffer_pool_start (GstBufferPool *base_pool)
{
    GstXCamBufferPool *pool = GST_XCAM_BUFFER_POOL (base_pool);
    XCAM_ASSERT (pool);
    SmartPtr<MainDeviceManager> device_manager = pool->device_manager;
    XCAM_ASSERT (device_manager.ptr ());
    device_manager->resume_dequeue ();
    return TRUE;
}

static gboolean
gst_xcam_buffer_pool_stop (GstBufferPool *base_pool)
{
    GstXCamBufferPool *pool = GST_XCAM_BUFFER_POOL (base_pool);
    XCAM_ASSERT (pool);
    SmartPtr<MainDeviceManager> device_manager = pool->device_manager;
    XCAM_ASSERT (device_manager.ptr ());

    device_manager->pause_dequeue ();
    return TRUE;
}

gboolean
gst_xcam_buffer_pool_set_config (GstBufferPool *base_pool, GstStructure *config)
{
    GstXCamBufferPool *pool = GST_XCAM_BUFFER_POOL (base_pool);

    XCAM_ASSERT (pool);
    pool->need_video_meta = gst_buffer_pool_config_has_option (config, GST_BUFFER_POOL_OPTION_VIDEO_META);

    pool->allocator = gst_dmabuf_allocator_new ();
    if (pool->allocator == NULL) {
        GST_WARNING ("xcam buffer pool get allocator failed");
        return FALSE;
    }

    return TRUE;
}

static GstFlowReturn
gst_xcam_buffer_pool_acquire_buffer (
    GstBufferPool *base_pool,
    GstBuffer **buffer,
    GstBufferPoolAcquireParams *params)
{
    GstXCamBufferPool *pool = GST_XCAM_BUFFER_POOL (base_pool);
    XCAM_ASSERT (pool);
    GstBuffer *out_buf = NULL;
    GstMemory *mem = NULL;
    GstXCamBufferMeta *meta = NULL;
    SmartPtr<MainDeviceManager> device_manager = pool->device_manager;
    SmartPtr<VideoBuffer> video_buf = device_manager->dequeue_buffer ();
    VideoBufferInfo video_info;
    gsize offsets[XCAM_VIDEO_MAX_COMPONENTS];

    XCAM_UNUSED (params);

    if (!video_buf.ptr ())
        return GST_FLOW_ERROR;

    video_info = video_buf->get_video_info ();
    for (int i = 0; i < XCAM_VIDEO_MAX_COMPONENTS; i++) {
        offsets[i] = video_info.offsets[i];
    }

    out_buf = gst_buffer_new ();
    meta = gst_buffer_add_xcam_buffer_meta (out_buf, video_buf);
    XCAM_ASSERT (meta);
    ((GstMeta *)(meta))->flags = (GstMetaFlags)(GST_META_FLAG_POOLED | GST_META_FLAG_LOCKED | GST_META_FLAG_READONLY);
    //GST_META_FLAG_SET (meta, (GST_META_FLAG_POOLED | GST_META_FLAG_LOCKED | GST_META_FLAG_READONLY));

    if (GST_XCAM_SRC_MEM_MODE (pool->src) == V4L2_MEMORY_DMABUF) {
        mem = gst_dmabuf_allocator_alloc (
                  pool->allocator, dup (video_buf->get_fd ()), video_buf->get_size ());
    } else if (GST_XCAM_SRC_MEM_MODE (pool->src) == V4L2_MEMORY_MMAP) {
        mem = gst_memory_new_wrapped (
                  (GstMemoryFlags)(GST_MEMORY_FLAG_READONLY | GST_MEMORY_FLAG_NO_SHARE),
                  video_buf->map (), video_buf->get_size (),
                  video_info.offsets[0], video_info.size,
                  NULL, NULL);
    } else {
        GST_WARNING ("xcam buffer pool acquire buffer failed since mem_type not supported");
        return GST_FLOW_ERROR;
    }

    XCAM_ASSERT (mem);
    gst_buffer_append_memory (out_buf, mem);
    if (pool->need_video_meta) {
        GstVideoMeta *video_meta =
            gst_buffer_add_video_meta_full (
                out_buf, GST_VIDEO_FRAME_FLAG_NONE,
                GST_VIDEO_INFO_FORMAT (GST_XCAM_SRC_OUT_VIDEO_INFO (pool->src)),
                video_info.width,
                video_info.height,
                video_info.components,
                offsets,
                (gint*)(video_info.strides));
        XCAM_ASSERT (video_meta);
        // TODO, consider map and unmap later
        video_meta->map = NULL;
        video_meta->unmap = NULL;
    }

    GST_BUFFER_TIMESTAMP (out_buf) = video_buf->get_timestamp () * 1000; //us to ns

    *buffer = out_buf;
    return GST_FLOW_OK;
}

static void
gst_xcam_buffer_pool_release_buffer (GstBufferPool *base_pool, GstBuffer *buffer)
{
    XCAM_UNUSED (base_pool);
    gst_buffer_unref (buffer);
}

GstBufferPool *
gst_xcam_buffer_pool_new (GstXCamSrc *src, GstCaps *caps, SmartPtr<MainDeviceManager> &device_manager)
{
    GstXCamBufferPool *pool;
    GstStructure *structure;

    pool = (GstXCamBufferPool *)g_object_new (GST_TYPE_XCAM_BUFFER_POOL, NULL);
    XCAM_ASSERT (pool);

    structure = gst_buffer_pool_get_config (GST_BUFFER_POOL_CAST (pool));
    XCAM_ASSERT (structure);
    gst_buffer_pool_config_set_params (
        structure, caps,
        GST_VIDEO_INFO_SIZE (GST_XCAM_SRC_OUT_VIDEO_INFO (src)),
        GST_XCAM_SRC_BUF_COUNT (src),
        GST_XCAM_SRC_BUF_COUNT (src));
    gst_buffer_pool_config_add_option (structure, GST_BUFFER_POOL_OPTION_VIDEO_META);
    gst_buffer_pool_set_config (GST_BUFFER_POOL_CAST (pool), structure);

    pool->src = src;
    gst_object_ref (src);
    pool->device_manager = device_manager;
    return GST_BUFFER_POOL (pool);
}
