/*
 * image_file_handle.h - Image file handle
 *
 *  Copyright (c) 2016 Intel Corporation
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

#ifndef XCAM_IMAGE_FILE_HANDLE_H
#define XCAM_IMAGE_FILE_HANDLE_H

#include "xcam_utils.h"
#include "buffer_pool.h"

namespace XCam {

class ImageFileHandle {
public:
    ImageFileHandle ()
        : _fp (NULL)
        , _file_name (NULL)
    {}
    ~ImageFileHandle ();

    bool is_valid () const {
        return (_fp ? true : false);
    }
    bool end_of_file ();
    XCamReturn open (const char *name, const char *option);
    XCamReturn close ();
    XCamReturn rewind ();
    XCamReturn read_buf (const SmartPtr<BufferProxy> &buf);
    XCamReturn write_buf (const SmartPtr<BufferProxy> buf);

private:
    XCAM_DEAD_COPY (ImageFileHandle);

private:
    FILE   *_fp;
    char   *_file_name;
};

}

#endif  //XCAM_IMAGE_FILE_HANDLE_H