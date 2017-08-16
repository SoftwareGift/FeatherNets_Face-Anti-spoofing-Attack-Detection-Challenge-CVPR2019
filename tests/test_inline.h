/*
 * test_inline.h - test inline header
 *
 *  Copyright (c) 2017 Intel Corporation
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
 *         Yinhang Liu <yinhangx.liu@intel.com>
 */

#ifndef XCAM_TEST_INLINE_H
#define XCAM_TEST_INLINE_H

#include "video_buffer.h"

using namespace XCam;

inline static void
ensure_gpu_buffer_done (SmartPtr<VideoBuffer> buf)
{
    const VideoBufferInfo info = buf->get_video_info ();
    VideoBufferPlanarInfo planar;
    uint8_t *memory = NULL;

    memory = buf->map ();
    for (uint32_t index = 0; index < info.components; index++) {
        info.get_planar_info (planar, index);
        uint32_t line_bytes = planar.width * planar.pixel_bytes;

        for (uint32_t i = 0; i < planar.height; i++) {
            int mem_idx = info.offsets [index] + i * info.strides [index] + line_bytes - 1;
            if (memory[mem_idx] == 1) {
                memory[mem_idx] = 1;
            }
        }
    }
    buf->unmap ();
}

class FileHandle {
public:
    FileHandle ()
        : _fp (NULL)
        , _file_name (NULL)
    {}
    ~FileHandle ();

    bool is_valid () const {
        return (_fp ? true : false);
    }
    bool end_of_file ();
    XCamReturn open (const char *name, const char *option);
    XCamReturn close ();
    XCamReturn rewind ();
    XCamReturn get_file_size (size_t &size);
    XCamReturn read_file (void *buf, const size_t &size);
    XCamReturn write_file (const void *buf, const size_t &size);

private:
    XCAM_DEAD_COPY (FileHandle);

private:
    FILE    *_fp;
    char    *_file_name;
};

FileHandle::~FileHandle ()
{
    close ();
}

bool
FileHandle::end_of_file()
{
    if (!is_valid ())
        return true;

    return feof (_fp);
}

XCamReturn
FileHandle::open (const char *name, const char *option)
{
    XCAM_ASSERT (name);
    if (!name)
        return XCAM_RETURN_ERROR_FILE;

    close ();
    XCAM_ASSERT (!_file_name && !_fp);
    _fp = fopen (name, option);

    if (!_fp)
        return XCAM_RETURN_ERROR_FILE;
    _file_name = strndup (name, 512);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
FileHandle::close ()
{
    if (_fp) {
        fclose (_fp);
        _fp = NULL;
    }

    if (_file_name) {
        xcam_free (_file_name);
        _file_name = NULL;
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
FileHandle::rewind ()
{
    if (!is_valid ())
        return XCAM_RETURN_ERROR_FILE;
    return (fseek (_fp, 0L, SEEK_SET) == 0) ? XCAM_RETURN_NO_ERROR : XCAM_RETURN_ERROR_FILE;
}

XCamReturn
FileHandle::get_file_size (size_t &size)
{
    if (fseek (_fp, 0L, SEEK_END) != 0)
        goto read_error;

    if ((size = ftell (_fp)) <= 0)
        goto read_error;

    if (fseek (_fp, 0L, SEEK_SET) != 0)
        goto read_error;

    return XCAM_RETURN_NO_ERROR;

read_error:
    XCAM_LOG_ERROR ("get file size failed");
    return XCAM_RETURN_ERROR_FILE;
}

XCamReturn
FileHandle::read_file (void *buf, const size_t &size)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    if (fread (buf, 1, size, _fp) != size) {
        XCAM_LOG_ERROR ("read file failed, size doesn't match");
        ret = XCAM_RETURN_ERROR_FILE;
    }

    return ret;
}

XCamReturn
FileHandle::write_file (const void *buf, const size_t &size)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    if (fwrite (buf, 1, size, _fp) != size) {
        XCAM_LOG_ERROR ("write file failed, size doesn't match");
        ret = XCAM_RETURN_ERROR_FILE;
    }

    return ret;
}

#endif // XCAM_TEST_INLINE_H
