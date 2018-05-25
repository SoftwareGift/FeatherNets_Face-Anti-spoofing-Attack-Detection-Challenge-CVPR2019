/*
 * gl_utils.h - GL utilities implementation
 *
 *  Copyright (c) 2018 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Author: Yinhang Liu <yinhangx.liu@intel.com>
 */

#include "gl_utils.h"

namespace XCam {

SmartPtr<GLBuffer> get_glbuffer (const SmartPtr<VideoBuffer> &buf)
{
    SmartPtr<GLVideoBuffer> gl_video_buf = buf.dynamic_cast_ptr<GLVideoBuffer> ();
    XCAM_FAIL_RETURN (
        ERROR, gl_video_buf.ptr (), NULL,
        "convert VideoBuffer to GLVideoBuffer failed");

    SmartPtr<GLBuffer> gl_buf = gl_video_buf->get_gl_buffer ();
    XCAM_ASSERT (gl_buf.ptr ());
    XCAM_FAIL_RETURN (
        ERROR, gl_buf.ptr (), NULL,
        "get GLBuffer from GLVideoBuffer failed");

    return gl_buf;
}

}
