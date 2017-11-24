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

#include <xcam_std.h>
#include <swapped_buffer.h>
#include <x3a_result.h>
#include <ocl/cl_kernel.h>
#include <ocl/cl_argument.h>
#include <ocl/cl_memory.h>
#include <ocl/cl_video_buffer.h>

namespace XCam {

class CLImageHandler;

class CLImageKernel
    : public CLKernel
{
    friend class CLImageHandler;

public:
    explicit CLImageKernel (const SmartPtr<CLContext> &context, const char *name = NULL, bool enable = true);
    virtual ~CLImageKernel ();

    void set_enable (bool enable) {
        _enable = enable;
    }

    bool is_enabled () const {
        return _enable;
    }
    virtual void pre_stop () {}

protected:
    XCamReturn pre_execute ();
    virtual XCamReturn prepare_arguments (
        CLArgList &args, CLWorkSize &work_size);

private:
    XCAM_DEAD_COPY (CLImageKernel);

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
        CLVideoPoolType = 0x0000,
        CLBoPoolType = 0x0001,
        DrmBoPoolType = 0x0002
    };

public:
    explicit CLImageHandler (const SmartPtr<CLContext> &context, const char *name);
    virtual ~CLImageHandler ();
    const char *get_name () const {
        return _name;
    }
    SmartPtr<CLContext> &get_context () {
        return  _context;
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

    bool add_kernel (const SmartPtr<CLImageKernel> &kernel);
    bool enable_handler (bool enable);
    bool is_handler_enabled () const;

    virtual bool is_ready ();
    XCamReturn execute (SmartPtr<VideoBuffer> &input, SmartPtr<VideoBuffer> &output);
    virtual void emit_stop ();

    SmartPtr<VideoBuffer> &get_input_buf ();
    SmartPtr<VideoBuffer> &get_output_buf ();

private:
    virtual XCamReturn prepare_buffer_pool_video_info (
        const VideoBufferInfo &input,
        VideoBufferInfo &output);

    // if derive prepare_output_buf, then prepare_buffer_pool_video_info is not involked
    virtual XCamReturn prepare_parameters (SmartPtr<VideoBuffer> &input, SmartPtr<VideoBuffer> &output);
    virtual XCamReturn execute_done (SmartPtr<VideoBuffer> &output);

protected:
    virtual XCamReturn prepare_output_buf (SmartPtr<VideoBuffer> &input, SmartPtr<VideoBuffer> &output);

    //only for multi-handler
    virtual XCamReturn execute_kernels ();

    XCamReturn ensure_parameters (SmartPtr<VideoBuffer> &input, SmartPtr<VideoBuffer> &output);
    XCamReturn execute_kernel (SmartPtr<CLImageKernel> &kernel);
    XCamReturn create_buffer_pool (const VideoBufferInfo &video_info);
    SmartPtr<BufferPool> &get_buffer_pool () {
        return _buf_pool;
    }
    void reset_buf_cache (const SmartPtr<VideoBuffer>& input, const SmartPtr<VideoBuffer>& output);

    bool append_kernels (SmartPtr<CLImageHandler> handler);

private:
    XCAM_DEAD_COPY (CLImageHandler);

private:
    char                      *_name;
    bool                       _enable;
    KernelList                 _kernels;
    SmartPtr<CLContext>        _context;
    SmartPtr<BufferPool>       _buf_pool;
    BufferPoolType             _buf_pool_type;
    bool                       _disable_buf_pool;
    uint32_t                   _buf_pool_size;
    uint32_t                   _buf_swap_flags;
    uint32_t                   _buf_swap_init_order;
    X3aResultList              _3a_results;
    int64_t                    _result_timestamp;

    SmartPtr<VideoBuffer>      _input_buf_cache;
    SmartPtr<VideoBuffer>      _output_buf_cache;

    XCAM_OBJ_PROFILING_DEFINES;
};

// never allocate buffer, only swap output from input
class CLCloneImageHandler
    : public CLImageHandler
{
public:
    explicit CLCloneImageHandler (const SmartPtr<CLContext> &context, const char *name);
    void set_clone_flags (uint32_t flags) {
        _clone_flags = flags;
    }
    uint32_t get_clone_flags () const {
        return _clone_flags;
    }

protected:
    //derived from CLImageHandler
    virtual XCamReturn prepare_output_buf (SmartPtr<VideoBuffer> &input, SmartPtr<VideoBuffer> &output);

private:
    XCAM_DEAD_COPY (CLCloneImageHandler);

    uint32_t                   _clone_flags;
};


};

#endif // XCAM_CL_IMAGE_HANDLER_H
