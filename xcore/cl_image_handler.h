/*
 * cl_image_handler.h - CL image handler
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

#ifndef XCAM_CL_IMAGE_HANDLER_H
#define XCAM_CL_IMAGE_HANDLER_H

#include "xcam_utils.h"
#include "cl_kernel.h"
#include "drm_bo_buffer.h"
#include "cl_memory.h"

namespace XCam {

#define XCAM_DEFAULT_IMAGE_DIM 2

class CLImageKernel
    : public CLKernel
{
public:
    explicit CLImageKernel (SmartPtr<CLContext> &context, const char *name);
    virtual ~CLImageKernel ();

    virtual XCamReturn pre_execute (SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output);
    virtual XCamReturn post_execute ();

private:
    XCAM_DEAD_COPY (CLImageKernel);

protected:
    SmartPtr<CLVaImage>   _image_in;
    SmartPtr<CLVaImage>   _image_out;
};

class CLImageHandler
{
    typedef std::list<SmartPtr<CLImageKernel>> KernelList;
public:
    explicit CLImageHandler (const char *name);
    virtual ~CLImageHandler ();
    const char *get_name () const {
        return _name;
    }

    bool add_kernel (SmartPtr<CLImageKernel> &kernel);
    virtual XCamReturn prepare_output_buf (SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output);
    XCamReturn execute (SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output);

protected:
    XCamReturn ensure_buffer_pool (const VideoBufferInfo &video_info);

private:
    XCAM_DEAD_COPY (CLImageHandler);

private:
    char                      *_name;
    KernelList                 _kernels;
    SmartPtr<DrmBoBufferPool>  _buf_pool;
};

};

#endif // XCAM_CL_IMAGE_HANDLER_H
