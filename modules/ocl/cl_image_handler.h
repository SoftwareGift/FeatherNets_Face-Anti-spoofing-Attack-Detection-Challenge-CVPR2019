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
#include "x3a_result.h"

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
    explicit CLImageKernel (SmartPtr<CLContext> &context, const char *name = NULL, bool enable = true);
    virtual ~CLImageKernel ();

    void set_enable (bool enable) {
        _enable = enable;
    }

    bool is_enabled () const {
        return _enable;
    }

    XCamReturn pre_execute (SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output);
    virtual XCamReturn post_execute (SmartPtr<DrmBoBuffer> &output);
    virtual void pre_stop () {}

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

private:
    bool                _enable;
};

class CLMultiImageHandler;
class CLImageHandler
{
    friend class CLMultiImageHandler;

public:
    typedef std::list<SmartPtr<CLImageKernel>> KernelList;
    enum BufferPoolType {
        CLBoPoolType  = 0x0001,
        DrmBoPoolType = 0x0002,
    };

public:
    explicit CLImageHandler (const char *name);
    virtual ~CLImageHandler ();
    const char *get_name () const {
        return _name;
    }

    void set_3a_result (SmartPtr<X3aResult> &result);
    SmartPtr<X3aResult> get_3a_result (XCam3aResultType type);

    int64_t get_result_timestamp () const {
        return _result_timestamp;
    };

    void set_pool_type (BufferPoolType type) {
        _buf_pool_type = type;
    }
    void set_pool_size (uint32_t size) {
        XCAM_ASSERT (size);
        _buf_pool_size = size;
    }
    void disable_buf_pool (bool flag) {
        _disable_buf_pool = flag;
    }

    bool is_buf_pool_disabled () const {
        return _disable_buf_pool;
    }

    bool enable_buf_pool_swap_flags (
        uint32_t flags,
        uint32_t init_order = (uint32_t)(SwappedBuffer::OrderY0Y1));

    bool add_kernel (SmartPtr<CLImageKernel> &kernel);
    bool enable_handler (bool enable);
    bool is_handler_enabled () const;

    virtual bool is_ready ();
    XCamReturn execute (SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output);
    virtual void emit_stop ();

protected:
    virtual XCamReturn prepare_buffer_pool_video_info (
        const VideoBufferInfo &input,
        VideoBufferInfo &output);

    // if derive prepare_output_buf, then prepare_buffer_pool_video_info is not involked
    virtual XCamReturn prepare_output_buf (SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output);
    virtual XCamReturn prepare_parameters (SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output);
    virtual XCamReturn execute_done (SmartPtr<DrmBoBuffer> &output);
    XCamReturn create_buffer_pool (const VideoBufferInfo &video_info);
    SmartPtr<BufferPool> &get_buffer_pool () {
        return _buf_pool;
    }

    bool append_kernels (SmartPtr<CLImageHandler> handler);

private:
    XCAM_DEAD_COPY (CLImageHandler);

private:
    char                      *_name;
    bool                       _enable;
    KernelList                 _kernels;
    SmartPtr<BufferPool>       _buf_pool;
    BufferPoolType             _buf_pool_type;
    bool                       _disable_buf_pool;
    uint32_t                   _buf_pool_size;
    uint32_t                   _buf_swap_flags;
    uint32_t                   _buf_swap_init_order;
    X3aResultList              _3a_results;
    int64_t                    _result_timestamp;

    XCAM_OBJ_PROFILING_DEFINES;
};

// never allocate buffer, only swap output from input
class CLCloneImageHandler
    : public CLImageHandler
{
public:
    explicit CLCloneImageHandler (const char *name);
    void set_clone_flags (uint32_t flags) {
        _clone_flags = flags;
    }
    uint32_t get_clone_flags () const {
        return _clone_flags;
    }

protected:
    //derived from CLImageHandler
    virtual XCamReturn prepare_output_buf (SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output);

private:
    XCAM_DEAD_COPY (CLCloneImageHandler);

    uint32_t                   _clone_flags;
};


};

#endif // XCAM_CL_IMAGE_HANDLER_H
