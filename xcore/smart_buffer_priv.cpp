/*
 * smart_buffer_priv.cpp - smart buffer for XCamVideoBuffer
 *
 *  Copyright (c) 2016-2017 Intel Corporation
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

#include <xcam_std.h>
#include "base/xcam_buffer.h"
#include "video_buffer.h"
#if HAVE_LIBDRM
#include "drm_bo_buffer.h"
#endif

namespace XCam {

class SmartBufferPriv
    : public XCamVideoBufferIntel
{
public:
    SmartBufferPriv (const SmartPtr<VideoBuffer> &buf);
    ~SmartBufferPriv ();

    bool is_valid () const {
        return _buf_ptr.ptr ();
    }

    static void     buf_ref (XCamVideoBuffer *data);
    static void     buf_unref (XCamVideoBuffer *data);
    static uint8_t *buf_map (XCamVideoBuffer *data);
    static void     buf_unmap (XCamVideoBuffer *data);
    static int      buf_get_fd (XCamVideoBuffer *data);
    static void    *buf_get_bo (XCamVideoBufferIntel *data);

private:
    XCAM_DEAD_COPY (SmartBufferPriv);

private:
    mutable RefCount       *_ref;
    SmartPtr<VideoBuffer>   _buf_ptr;
};

SmartBufferPriv::SmartBufferPriv (const SmartPtr<VideoBuffer> &buf)
    : _ref (NULL)
{
    XCAM_ASSERT (buf.ptr ());
    this->_buf_ptr = buf;

    if (!buf.ptr ()) {
        return;
    }

    _ref = new RefCount ();

    const VideoBufferInfo& video_info = buf->get_video_info ();

    this->base.info = *((const XCamVideoBufferInfo*)&video_info);
    this->base.mem_type = XCAM_MEM_TYPE_PRIVATE_BO;
    this->base.timestamp = buf->get_timestamp ();

    this->base.ref = SmartBufferPriv::buf_ref;
    this->base.unref = SmartBufferPriv::buf_unref;
    this->base.map = SmartBufferPriv::buf_map;
    this->base.unmap = SmartBufferPriv::buf_unmap;
    this->base.get_fd = SmartBufferPriv::buf_get_fd;
    this->get_bo = SmartBufferPriv::buf_get_bo;
}

SmartBufferPriv::~SmartBufferPriv ()
{
    delete _ref;
}

void
SmartBufferPriv::buf_ref (XCamVideoBuffer *data)
{
    SmartBufferPriv *buf = (SmartBufferPriv*) data;
    XCAM_ASSERT (buf->_ref);
    if (buf->_ref)
        buf->_ref->ref ();
}

void
SmartBufferPriv::buf_unref (XCamVideoBuffer *data)
{
    SmartBufferPriv *buf = (SmartBufferPriv*) data;
    XCAM_ASSERT (buf->_ref);
    if (buf->_ref) {
        if (!buf->_ref->unref()) {
            delete buf;
        }
    }
}

uint8_t *
SmartBufferPriv::buf_map (XCamVideoBuffer *data)
{
    SmartBufferPriv *buf = (SmartBufferPriv*) data;
    XCAM_ASSERT (buf->_buf_ptr.ptr ());
    return buf->_buf_ptr->map ();
}

void
SmartBufferPriv::buf_unmap (XCamVideoBuffer *data)
{
    SmartBufferPriv *buf = (SmartBufferPriv*) data;
    XCAM_ASSERT (buf->_buf_ptr.ptr ());
    buf->_buf_ptr->unmap ();
}

int
SmartBufferPriv::buf_get_fd (XCamVideoBuffer *data)
{
    SmartBufferPriv *buf = (SmartBufferPriv*) data;
    XCAM_ASSERT (buf->_buf_ptr.ptr ());
    return buf->_buf_ptr->get_fd ();
}

void *
SmartBufferPriv::buf_get_bo (XCamVideoBufferIntel *data)
{
#if HAVE_LIBDRM
    SmartBufferPriv *buf = (SmartBufferPriv*) data;
    XCAM_ASSERT (buf->_buf_ptr.ptr ());

    SmartPtr<DrmBoBuffer> bo_buf = buf->_buf_ptr.dynamic_cast_ptr<DrmBoBuffer> ();
    XCAM_FAIL_RETURN (
        ERROR,
        bo_buf.ptr (),
        NULL,
        "get DrmBoBuffer failed");

    return bo_buf->get_bo ();
#else
    XCAM_LOG_ERROR ("VideoBuffer doesn't support DrmBoBuffer");

    XCAM_UNUSED (data);
    return NULL;
#endif
}

XCamVideoBuffer *
convert_to_external_buffer (const SmartPtr<VideoBuffer> &buf)
{
    SmartBufferPriv *priv_buf = new SmartBufferPriv (buf);
    XCAM_ASSERT (priv_buf);

    if (priv_buf->is_valid ())
        return (XCamVideoBuffer *)(priv_buf);

    delete priv_buf;
    return NULL;
}

}
