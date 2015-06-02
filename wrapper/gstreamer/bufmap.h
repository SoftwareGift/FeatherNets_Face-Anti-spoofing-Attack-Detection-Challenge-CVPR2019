/*
 * bufmap.h - map V4l2Buffer to GstBuffer
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

#ifndef __BUFMAP_H__
#define __BUFMAP_H__

#include <assert.h>
#include <gst/gst.h>
#include <map>
#include "xcam_defs.h"
#include "smartptr.h"
#include "xcam_mutex.h"
#include "video_buffer.h"
#include "v4l2_buffer_proxy.h"

namespace XCam {

class BufMap {
public:
    static SmartPtr<BufMap> instance();

    GstBuffer* gbuf(SmartPtr<VideoBuffer> &buf) {
        XCAM_ASSERT (buf.ptr ());
        if (buf->get_fd ()) {
            // DMA buffer
            if (_v2g.find (buf->get_fd ()) == _v2g.end ()) { //non-existing
                return NULL;
            }
            return _v2g[buf->get_fd ()];
        }
        else {
            // V4L2 buffer
            SmartPtr<V4l2BufferProxy> v4l2_buf = buf.dynamic_cast_ptr<V4l2BufferProxy> ();
            if (_v2g.find (v4l2_buf->get_v4l2_buf_index ()) == _v2g.end ()) { //non-existing
                return NULL;
            }
            return _v2g[v4l2_buf->get_v4l2_buf_index ()];
        }
    }
    int vbuf(GstBuffer* gbuf) {
        if (_g2v.find (gbuf) == _g2v.end ()) { //non-existing
            return NULL;
        }
        return _g2v[gbuf];
    }
    void setmap(GstBuffer* gbuf, SmartPtr<VideoBuffer>& buf) {
        XCAM_ASSERT (buf.ptr ());
        if (buf->get_fd ()) {
            // DMA buffer
            _g2v[gbuf] = buf->get_fd ();
            _v2g[buf->get_fd ()] = gbuf;
        }
        else {
            // V4L2 buffer
            SmartPtr<V4l2BufferProxy> v4l2_buf = buf.dynamic_cast_ptr<V4l2BufferProxy> ();
            _g2v[gbuf] = v4l2_buf->get_v4l2_buf_index ();
            _v2g[v4l2_buf->get_v4l2_buf_index ()] = gbuf;
        }
    }

private:
    XCAM_DEAD_COPY (BufMap);

private:
    BufMap() {};

    static SmartPtr<BufMap> _instance;
    static Mutex        _mutex;

    std::map <GstBuffer*, int> _g2v;
    std::map <int, GstBuffer*> _v2g;
};

} //namespace

#endif // __BUFMAP_H__
