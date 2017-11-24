/*
 * cl_kernel.h - CL kernel
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

#ifndef XCAM_CL_KERNEL_H
#define XCAM_CL_KERNEL_H

#include <xcam_std.h>
#include <xcam_mutex.h>
#include <ocl/cl_event.h>
#include <ocl/cl_argument.h>

#include <CL/cl.h>
#include <string>
#include <unistd.h>

#define XCAM_CL_KERNEL_FUNC_SOURCE_BEGIN(func)  \
    const char func##_body []=
//const char *func##_name = #func;

#define XCAM_CL_KERNEL_FUNC_BINARY_BEGIN(func)  \
    const uint8_t func##_body[] =

#define XCAM_CL_KERNEL_FUNC_END

XCAM_BEGIN_DECLARE

typedef struct _XCamKernelInfo {
    const char   *kernel_name;
    const char   *kernel_body;
    size_t        kernel_body_len;
} XCamKernelInfo;

XCAM_END_DECLARE

namespace XCam {

class CLContext;
class CLKernel;

/*
 * Example to create a kernel
 * XCAM_CL_KERNEL_FUNC_SOURCE_BEGIN(kernel_demo)
 * #include "kernel_demo.clx"
 * XCAM_CL_KERNEL_FUNC_END
 * SmartPtr<CLKernel> kernel = new CLKernel (context, kernel_demo);
 * kernel->load_from_source (kernel_demo_body, strlen(kernel_demo_body));
 * XCAM_ASSERT (kernel->is_valid());
 */
class CLKernel {
    friend class CLContext;
public:
    explicit CLKernel (const SmartPtr<CLContext> &context, const char *name);
    virtual ~CLKernel ();

    XCamReturn build_kernel (const XCamKernelInfo& info, const char* options = NULL);

    cl_kernel get_kernel_id () {
        return _kernel_id;
    }
    bool is_valid () const {
        return _kernel_id != NULL;
    }
    const char *get_kernel_name () const {
        return _name;
    }
    SmartPtr<CLContext> &get_context () {
        return  _context;
    }

    XCamReturn set_arguments (const CLArgList &args, const CLWorkSize &work_size);
    const CLWorkSize &get_work_size () const {
        return _work_size;
    }

    bool is_arguments_set () const {
        return !_arg_list.empty ();
    }
    const CLArgList &get_args () const {
        return _arg_list;
    }

    XCamReturn execute (
        const SmartPtr<CLKernel> self,
        bool block = false,
        CLEventList &events = CLEvent::EmptyList,
        SmartPtr<CLEvent> &event_out = CLEvent::NullEvent);

    XCamReturn load_from_source (
        const char *source, size_t length = 0,
        uint8_t **gen_binary = NULL,
        size_t *binary_size = NULL,
        const char *build_option = NULL);

    XCamReturn load_from_binary (const uint8_t *binary, size_t length);

private:
    XCamReturn set_argument (uint32_t arg_i, void *arg_addr, uint32_t arg_size);
    XCamReturn set_work_size (const CLWorkSize &work_size);
    void set_default_work_size ();
    void destroy ();
    XCamReturn clone (SmartPtr<CLKernel> kernel);

    static void event_notify (cl_event event, cl_int status, void* data);
    XCAM_DEAD_COPY (CLKernel);

private:
    typedef std::map<std::string, SmartPtr<CLKernel> > KernelMap;

    static KernelMap      _kernel_map;
    static Mutex          _kernel_map_mutex;
    static const char    *_kernel_cache_path;

private:
    char                 *_name;
    cl_kernel             _kernel_id;
    SmartPtr<CLContext>   _context;
    SmartPtr<CLKernel>    _parent_kernel;
    CLArgList             _arg_list;
    CLWorkSize            _work_size;

    XCAM_OBJ_PROFILING_DEFINES;
};

};

#endif //XCAM_CL_KERNEL_H
