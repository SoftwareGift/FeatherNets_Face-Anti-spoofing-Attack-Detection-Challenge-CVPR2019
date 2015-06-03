/*
 * gstxcambuffermeta.cpp - gst xcam buffer meta data
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

#include "gstxcambuffermeta.h"

GType
gst_xcam_buffer_meta_api_get_type (void)
{
    static GType xcam_buf_type = 0;
    static const gchar *xcam_buf_tags [] =
    { GST_XCAM_META_TAG_XCAM, GST_XCAM_META_TAG_BUF, NULL };

    if (g_once_init_enter (&xcam_buf_type)) {
        GType _type = gst_meta_api_type_register ("GstXCamBuffer", xcam_buf_tags);
        g_once_init_leave (&xcam_buf_type, _type);
    }

    return xcam_buf_type;
}

static gboolean
gst_xcam_buffer_meta_init (GstMeta *base, gpointer params, GstBuffer *buffer)
{
    XCAM_UNUSED (params);
    XCAM_UNUSED (buffer);
    GstXCamBufferMeta *meta = (GstXCamBufferMeta *)base;

    XCAM_CONSTRUCTOR (meta->buffer, SmartPtr<VideoBuffer>);
    return TRUE;
}


static void
gst_xcam_buffer_meta_free (GstMeta *base, GstBuffer *buffer)
{
    XCAM_UNUSED (buffer);
    GstXCamBufferMeta *meta = (GstXCamBufferMeta *)base;

    meta->buffer->unmap ();
    XCAM_DESTRUCTOR (meta->buffer, SmartPtr<VideoBuffer>);
}

static const GstMetaInfo *
gst_xcam_buffer_meta_get_info (void)
{
    static const GstMetaInfo *meta_info = NULL;

    if (g_once_init_enter (&meta_info)) {
        const GstMetaInfo *_meta =
            gst_meta_register (GST_XCAM_BUFFER_META_API_TYPE,
                               "GstXCamBufferMeta",
                               sizeof (GstXCamBufferMeta),
                               gst_xcam_buffer_meta_init,
                               gst_xcam_buffer_meta_free,
                               NULL);
        g_once_init_leave (&meta_info, _meta);
    }
    return meta_info;
}

GstXCamBufferMeta *
gst_buffer_add_xcam_buffer_meta (
    GstBuffer *buffer,
    const SmartPtr<VideoBuffer>  &data)
{
    XCAM_ASSERT (data.ptr ());

    GstXCamBufferMeta *meta = (GstXCamBufferMeta*) gst_buffer_add_meta (
                                  buffer, gst_xcam_buffer_meta_get_info(), NULL);

    g_return_val_if_fail (meta, NULL);

    meta->buffer = data;

    return meta;
}


