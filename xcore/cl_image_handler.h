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

struct CLWorkSize
{
    uint32_t dim;
    size_t global[XCAM_CL_KERNEL_MAX_WORK_DIM];
    size_t local[XCAM_CL_KERNEL_MAX_WORK_DIM];
    CLWorkSize();
};

struct CLArgument
{
    void     *arg_adress;
    uint32_t  arg_size;
    CLArgument ();
};

class CLImageKernel
    : public CLKernel
{
public:
    explicit CLImageKernel (SmartPtr<CLContext> &context, const char *name);
    virtual ~CLImageKernel ();

    XCamReturn pre_execute (SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output);
    virtual XCamReturn post_execute ();

protected:
    virtual XCamReturn prepare_arguments (
        SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
        CLArgument args[], uint32_t &arg_count,
        CLWorkSize &work_size);

private:
    XCAM_DEAD_COPY (CLImageKernel);

protected:
    SmartPtr<CLImage>   _image_in;
    SmartPtr<CLImage>   _image_out;
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
    XCamReturn execute (SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output);
    void emit_stop ();

protected:
    virtual XCamReturn prepare_buffer_pool_video_info (
        const VideoBufferInfo &input,
        VideoBufferInfo &output);

    // if derive prepare_output_buf, then prepare_buffer_pool_video_info is not involked
    virtual XCamReturn prepare_output_buf (SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output);
    XCamReturn create_buffer_pool (const VideoBufferInfo &video_info);
    SmartPtr<BufferPool> &get_buffer_pool () {
        return _buf_pool;
    }

private:
    XCAM_DEAD_COPY (CLImageHandler);

private:
    char                      *_name;
    KernelList                 _kernels;
    SmartPtr<BufferPool>       _buf_pool;
};

};

#endif // XCAM_CL_IMAGE_HANDLER_H
