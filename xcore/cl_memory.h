/*
 * cl_memory.h - CL memory
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
 * Author: Wind Yuan <feng.yuan@intel.com>
 */

#ifndef XCAM_CL_MEMORY_H
#define XCAM_CL_MEMORY_H

#include "xcam_utils.h"
#include "cl_context.h"
#include "drm_bo_buffer.h"

namespace XCam {

class CLMemory {
public:
    CLMemory (SmartPtr<CLContext> &context);
    virtual ~CLMemory ();

    cl_mem &get_mem_id () {
        return _mem_id;
    }
    bool is_valid () const {
        return _mem_id != NULL;
    }

private:
    XCAM_DEAD_COPY (CLMemory);

protected:
    SmartPtr<CLContext>   _context;
    cl_mem                _mem_id;
};

class CLVaImage
    : public CLMemory
{
public:
    explicit CLVaImage (
        SmartPtr<CLContext> &context,
        SmartPtr<DrmBoBuffer> &bo,
        const cl_libva_image *image_info = NULL);
    ~CLVaImage () {}
    const cl_libva_image & get_image_info () const {
        return _image_info;
    }

private:
    XCAM_DEAD_COPY (CLVaImage);
private:
    SmartPtr<DrmBoBuffer>   _bo;
    cl_libva_image          _image_info;
};

};
#endif //
