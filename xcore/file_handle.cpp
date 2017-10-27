/*
 * file_handle.cpp - File handle
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
 * Author: Yinhang Liu <yinhangx.liu@intel.com>
 * Author: Wind Yuan <feng.yuan@intel.com>
 */

#include "file_handle.h"

#define INVALID_SIZE (size_t)(-1)

namespace XCam {

FileHandle::FileHandle ()
    : _fp (NULL)
    , _file_name (NULL)
    , _file_size (INVALID_SIZE)
{}

FileHandle::FileHandle (const char *name, const char *option)
    : _fp (NULL)
    , _file_name (NULL)
    , _file_size (INVALID_SIZE)
{
    open (name, option);
}

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

    _file_size = INVALID_SIZE;
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
    if (_file_size != INVALID_SIZE) {
        size = _file_size;
        return XCAM_RETURN_NO_ERROR;
    }

    fpos_t cur_pos;
    long file_size;

    if (fgetpos (_fp, &cur_pos) < 0)
        goto read_error;

    if (fseek (_fp, 0L, SEEK_END) != 0)
        goto read_error;

    if ((file_size = ftell (_fp)) <= 0)
        goto read_error;

    if (fsetpos (_fp, &cur_pos) < 0)
        goto read_error;

    _file_size = file_size;
    size = file_size;
    return XCAM_RETURN_NO_ERROR;

read_error:
    XCAM_LOG_ERROR ("get file size failed with errno:%d", errno);
    return XCAM_RETURN_ERROR_FILE;
}

XCamReturn
FileHandle::read_file (void *buf, const size_t &size)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    if (fread (buf, 1, size, _fp) != size) {
        if (end_of_file ()) {
            ret = XCAM_RETURN_BYPASS;
        } else {
            XCAM_LOG_ERROR ("read file failed, size doesn't match");
            ret = XCAM_RETURN_ERROR_FILE;
        }
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

}
