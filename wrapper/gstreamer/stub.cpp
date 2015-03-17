/*
 * stub.cpp - stub utilities that implemented in CPP
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

#include "stub.h"
#include "bufmap.h"
#include "v4l2dev.h"

#include <stdio.h>

using namespace XCam;

int libxcam_set_device_name (const char *ch)
{
    V4l2Dev::_device_name = ch;
    return 0;
}

int libxcam_set_sensor_id (int id)
{
    SmartPtr<V4l2Device> device = V4l2Dev::instance();
    return (int) device->set_sensor_id (id);
}
int libxcam_set_capture_mode (uint32_t cap_mode)
{
    SmartPtr<V4l2Device> device = V4l2Dev::instance();
    return (int) device->set_capture_mode (cap_mode);
}
int libxcam_set_mem_type (enum v4l2_memory mem_type)
{
    SmartPtr<V4l2Device> device = V4l2Dev::instance();
    return (int) device->set_mem_type (mem_type);
}
int libxcam_set_buffer_count (uint32_t buf_count)
{
    SmartPtr<V4l2Device> device = V4l2Dev::instance();
    return (int) device->set_buffer_count (buf_count);
}
int libxcam_set_framerate (uint32_t fps_n, uint32_t fps_d)
{
    SmartPtr<V4l2Device> device = V4l2Dev::instance();
    return (int) device->set_framerate (fps_n, fps_d);
}
int libxcam_open ()
{
    SmartPtr<V4l2Device> device = V4l2Dev::instance();
    return (int) device->open ();
}
int libxcam_close ()
{
    SmartPtr<V4l2Device> device = V4l2Dev::instance();
    return (int) device->close ();
}
int libxcam_set_format (uint32_t width, uint32_t height, uint32_t pixelformat, enum v4l2_field field, uint32_t bytes_perline)
{
    SmartPtr<V4l2Device> device = V4l2Dev::instance();
    return (int) device->set_format (width, height, pixelformat, field, bytes_perline);
}
int libxcam_get_blocksize (uint32_t *blocksize)
{
    SmartPtr<V4l2Device> device = V4l2Dev::instance();
    struct v4l2_format format;
    int ret ;
    ret = (int) device->get_format (format);
    *blocksize = format.fmt.pix.sizeimage;
    return ret;
}
int libxcam_start ()
{
    SmartPtr<V4l2Device> device = V4l2Dev::instance();
    struct v4l2_format format;
    device->get_format (format);
#if HAVE_LIBDRM
    AtomispDevice *atom_isp_dev = (AtomispDevice *) device.ptr();
    SmartPtr<DrmDisplay> drmdisp = DrmDisplay::instance();
    struct v4l2_rect rect = {0, 0, (int)format.fmt.pix.width, (int)format.fmt.pix.height};
    drmdisp->drm_init (&format.fmt.pix,
                       "i915",
                       9,
                       3,
                       1920,
                       1080,
                       format.fmt.pix.pixelformat,
                       device->get_capture_buf_type(),
                       &rect);
    atom_isp_dev->set_drm_display (drmdisp);
#endif
    return (int) device->start();
}

int libxcam_stop ()
{
    SmartPtr<V4l2Device> device = V4l2Dev::instance();
    return (int) device->stop();
}

int libxcam_dequeue_buffer (SmartPtr<V4l2Buffer> &buf)
{
    SmartPtr<V4l2Device> device = V4l2Dev::instance();
    int ret;

    ret = device->poll_event (5000);
    if (ret < 0) {
        printf ("device(%s) poll event failed\n", device->get_device_name());
        return -1;
    } else if (ret == 0) {
        printf ("device(%s) poll event did not detect events\n", device->get_device_name());
        return -1;
    }

    ret = device->dequeue_buffer (buf);
    if (ret != XCAM_RETURN_NO_ERROR) {
        printf ("device(%s) dequeue buffer failed\n", device->get_device_name() );
        return ret;
    }

    return (int) XCAM_RETURN_NO_ERROR;
}
int libxcam_enqueue_buffer (SmartPtr<V4l2Buffer> &buf)
{
    SmartPtr<V4l2Device> device = V4l2Dev::instance();
    int ret;

    ret = device->queue_buffer (buf);
    if (ret != XCAM_RETURN_NO_ERROR) {
        printf ("device(%s) queue buffer failed\n", device->get_device_name());
        return ret;
    }
    return (int) XCAM_RETURN_NO_ERROR;
}

// FIXME remove the following 4 functions
GstBuffer* bufmap_2gbuf(SmartPtr<V4l2Buffer> &buf)
{
    SmartPtr<BufMap> bufmap = BufMap::instance();
    return bufmap->gbuf(buf);
}

SmartPtr<V4l2Buffer> bufmap_2vbuf(GstBuffer* gbuf)
{
    SmartPtr<BufMap> bufmap = BufMap::instance();
    return bufmap->vbuf(gbuf);
}

void bufmap_setmap(GstBuffer* gbuf, SmartPtr<V4l2Buffer> buf)
{
    SmartPtr<BufMap> bufmap = BufMap::instance();
    bufmap->setmap(gbuf, buf);
}

GstFlowReturn
xcam_bufferpool_acquire_buffer (GstBufferPool *bpool, GstBuffer **buffer, GstBufferPoolAcquireParams *params)
{
    GstBuffer *gbuf = NULL;
    Gstxcambufferpool *pool = GST_XCAMBUFFERPOOL_CAST (bpool);
    Gstxcamsrc *xcamsrc = pool->src;

    SmartPtr<V4l2Buffer> buf;
    libxcam_dequeue_buffer (buf);

    struct v4l2_buffer vbuf = buf->get_buf();

    gbuf = bufmap_2gbuf(buf);
    if (!gbuf) {
        gbuf = gst_buffer_new();
        GST_BUFFER (gbuf)->pool = (GstBufferPool*) pool;

        gst_buffer_append_memory (gbuf,
                                  gst_dmabuf_allocator_alloc (pool->allocator, vbuf.m.fd, vbuf.length));
        bufmap_setmap(gbuf, buf);
    }

    GST_BUFFER_TIMESTAMP (gbuf) = GST_TIMEVAL_TO_TIME (vbuf.timestamp);
    *buffer = gbuf;

    return GST_FLOW_OK;
}

void
xcambufferpool_release_buffer (GstBufferPool *bpool, GstBuffer *gbuf)
{
    SmartPtr<V4l2Buffer> buf = bufmap_2vbuf(gbuf);
    libxcam_enqueue_buffer (buf);
}


