/*
 * gl_video_buffer.h - GL video buffer class
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

#ifndef XCAM_GL_VIDEO_BUFFER_H
#define XCAM_GL_VIDEO_BUFFER_H

#include <buffer_pool.h>
#include <gles/gl_buffer.h>

namespace XCam {

class GLVideoBuffer
    : public BufferProxy
{
    friend class GLVideoBufferPool;

public:
    virtual ~GLVideoBuffer () {}
    SmartPtr<GLBuffer> get_gl_buffer ();

protected:
    explicit GLVideoBuffer (const VideoBufferInfo &info, const SmartPtr<BufferData> &data);
};

class GLVideoBufferPool
    : public BufferPool
{
public:
    explicit GLVideoBufferPool ();
    explicit GLVideoBufferPool (const VideoBufferInfo &info, GLenum target = GL_SHADER_STORAGE_BUFFER);
    virtual ~GLVideoBufferPool ();

    void set_binding_target (GLenum target) {
        _target = target;
    }

private:
    virtual SmartPtr<BufferData> allocate_data (const VideoBufferInfo &info);
    virtual SmartPtr<BufferProxy> create_buffer_from_data (SmartPtr<BufferData> &data);

private:
    GLenum    _target;
};

};
#endif // XCAM_GL_VIDEO_BUFFER_H