/*
 * gl_buffer.h - GL buffer
 *
 *  Copyright (c) 2018 Intel Corporation
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

#ifndef XCAM_GL_BUFFER_H
#define XCAM_GL_BUFFER_H

#include <gles/gles_std.h>
#include <map>

namespace XCam {

class GLBuffer
{
public:
    ~GLBuffer ();
    static SmartPtr<GLBuffer> create_buffer (
        GLenum target, const GLvoid *data = NULL, uint32_t size = 0, GLenum usage = GL_STATIC_DRAW);

    GLuint get_buffer_id () const {
        return _buf_id;
    }
    GLenum get_target () const {
        return _target;
    }
    GLenum get_usage () const {
        return _usage;
    }
    uint32_t get_size () const {
        return _size;
    }

    void *map_range (
        uint32_t offset = 0, uint32_t length = 0,
        GLbitfield flags = GL_MAP_READ_BIT | GL_MAP_WRITE_BIT);
    XCamReturn flush_map ();
    XCamReturn unmap ();

    XCamReturn bind ();
    XCamReturn bind_buffer_base (uint32_t index);
    XCamReturn bind_buffer_range (uint32_t index, uint32_t offset, uint32_t size);

private:
    explicit GLBuffer (GLuint id, GLenum type, GLenum usage, uint32_t size);

private:
    XCAM_DEAD_COPY (GLBuffer);

    struct MapRange {
        uint32_t    offset;
        uint32_t    len;
        GLbitfield  flags;
        void       *ptr;

        MapRange ();
        bool is_mapped () const;
    };

private:
    GLenum        _target;
    GLenum        _usage;
    GLuint        _buf_id;
    uint32_t      _size;
    MapRange      _mapped_range;
};

}

#endif  //XCAM_GL_BUFFER_H
