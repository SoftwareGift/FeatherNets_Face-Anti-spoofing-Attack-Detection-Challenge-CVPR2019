/*
 * image_file_handle.cpp - Image file handle
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
 * Author: Yinhang Liu <yinhangx.liu@intel.com>
 */

#include "image_file_handle.h"

namespace XCam {

ImageFileHandle::ImageFileHandle ()
{
}

ImageFileHandle::ImageFileHandle (const char *name, const char *option)
    : FileHandle (name, option)
{
}

ImageFileHandle::~ImageFileHandle ()
{
    close ();
}

XCamReturn
ImageFileHandle::read_buf (const SmartPtr<VideoBuffer> &buf)
{
    const VideoBufferInfo info = buf->get_video_info ();
    VideoBufferPlanarInfo planar;
    uint8_t *memory = NULL;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    XCAM_ASSERT (is_valid ());

    memory = buf->map ();
    for (uint32_t index = 0; index < info.components; index++) {
        info.get_planar_info (planar, index);
        uint32_t line_bytes = planar.width * planar.pixel_bytes;

        for (uint32_t i = 0; i < planar.height; i++) {
            if (fread (memory + info.offsets [index] + i * info.strides [index], 1, line_bytes, _fp) != line_bytes) {
                if (end_of_file ())
                    ret = XCAM_RETURN_BYPASS;
                else {
                    XCAM_LOG_ERROR ("read file failed, size doesn't match");
                    ret = XCAM_RETURN_ERROR_FILE;
                }
            }
        }
    }
    buf->unmap ();
    return ret;
}

XCamReturn
ImageFileHandle::write_buf (const SmartPtr<VideoBuffer> &buf)
{
    const VideoBufferInfo info = buf->get_video_info ();
    VideoBufferPlanarInfo planar;
    uint8_t *memory = NULL;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    XCAM_ASSERT (is_valid ());

    memory = buf->map ();
    for (uint32_t index = 0; index < info.components; index++) {
        info.get_planar_info (planar, index);
        uint32_t line_bytes = planar.width * planar.pixel_bytes;

        for (uint32_t i = 0; i < planar.height; i++) {
            if (fwrite (memory + info.offsets [index] + i * info.strides [index], 1, line_bytes, _fp) != line_bytes) {
                XCAM_LOG_ERROR ("write file failed, size doesn't match");
                ret = XCAM_RETURN_ERROR_FILE;
            }
        }
    }
    buf->unmap ();
    return ret;
}

}
