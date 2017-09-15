/*
 * cl_utils.cpp - CL Utilities
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

#include "cl_utils.h"

namespace XCam {

bool
write_image (SmartPtr<CLImage> image, const char *file_name)
{
    XCAM_ASSERT (file_name);

    const CLImageDesc &desc = image->get_image_desc ();
    void *ptr = NULL;
    size_t origin[3] = {0, 0, 0};
    size_t region[3] = {desc.width, desc.height, 1};
    size_t row_pitch;
    size_t slice_pitch;

    XCamReturn ret = image->enqueue_map (ptr, origin, region, &row_pitch, &slice_pitch, CL_MEM_READ_ONLY);
    XCAM_ASSERT (ret == XCAM_RETURN_NO_ERROR);
    XCAM_ASSERT (ptr);
    XCAM_ASSERT (row_pitch == desc.row_pitch);
    uint8_t *buf_start = (uint8_t *)ptr;
    uint32_t width = image->get_pixel_bytes () * desc.width;

    FILE *fp = fopen (file_name, "wb");
    XCAM_FAIL_RETURN (ERROR, fp, false, "open file(%s) failed", file_name);

    for (uint32_t i = 0; i < desc.height; ++i) {
        uint8_t *buf_line = buf_start + row_pitch * i;
        fwrite (buf_line, width, 1, fp);
    }
    image->enqueue_unmap (ptr);
    fclose (fp);
    XCAM_LOG_INFO ("write image:%s\n", file_name);
    return true;
}

SmartPtr<CLBuffer>
convert_to_clbuffer (
    const SmartPtr<CLContext> &context,
    SmartPtr<VideoBuffer> &buf)
{
    SmartPtr<CLBuffer> cl_buf;

    SmartPtr<DrmBoBuffer> bo_buf = buf.dynamic_cast_ptr<DrmBoBuffer> ();
    cl_buf = new CLVaBuffer (context, bo_buf);

    XCAM_FAIL_RETURN (WARNING, cl_buf.ptr (), NULL, "convert to clbuffer failed");
    return cl_buf;
}

SmartPtr<CLImage>
convert_to_climage (
    const SmartPtr<CLContext> &context,
    SmartPtr<VideoBuffer> &buf,
    const CLImageDesc &desc,
    uint32_t offset,
    cl_mem_flags flags)
{
    SmartPtr<CLImage> cl_image;

    SmartPtr<DrmBoBuffer> bo_buf = buf.dynamic_cast_ptr<DrmBoBuffer> ();
    cl_image = new CLVaImage (context, bo_buf, desc, offset);

    XCAM_FAIL_RETURN (WARNING, cl_image.ptr (), NULL, "convert to climage failed");
    return cl_image;
}

}
