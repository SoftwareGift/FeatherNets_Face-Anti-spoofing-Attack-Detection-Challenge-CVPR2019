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

#include "xcam_utils.h"
#include "smartptr.h"
#include "cl_event.h"
#include <CL/cl.h>
#include <string>


#define XCAM_CL_KERNEL_FUNC_SOURCE_BEGIN(func)  \
    const char func##_body []=
//const char *func##_name = #func;

#define XCAM_CL_KERNEL_FUNC_BINARY_BEGIN(func)  \
    const uint8_t func##_body[] =

#define XCAM_CL_KERNEL_FUNC_END

#define XCAM_CL_KERNEL_MAX_WORK_DIM 3

namespace XCam {

class CLContext;

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
    explicit CLKernel (SmartPtr<CLContext> &context, const char *name);
    virtual ~CLKernel ();

    XCamReturn load_from_source (
        const char *source, size_t length = 0,
        uint8_t **program_binaries = NULL,
        size_t *binary_sizes = NULL);
    XCamReturn load_from_binary (const uint8_t *binary, size_t length);
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

    XCamReturn set_argument (uint32_t arg_i, void *arg_addr, uint32_t arg_size);
    XCamReturn set_work_size (uint32_t dim, size_t *global, size_t *local);

    uint32_t get_work_dims () const {
        return _work_dim;
    }
    const size_t *get_work_global_size () const {
        return _global_work_size;
    }
    const size_t *get_work_local_size () const {
        return _local_work_size;
    }

    XCamReturn execute (
        CLEventList &events = CLEvent::EmptyList,
        SmartPtr<CLEvent> &event_out = CLEvent::NullEvent);

private:
    void set_default_work_size ();
    void destroy ();
    XCAM_DEAD_COPY (CLKernel);

private:
    char                 *_name;
    cl_kernel             _kernel_id;
    SmartPtr<CLContext>   _context;
    uint32_t              _work_dim;
    size_t                _global_work_size [XCAM_CL_KERNEL_MAX_WORK_DIM];
    size_t                _local_work_size [XCAM_CL_KERNEL_MAX_WORK_DIM];
};

};

#endif //XCAM_CL_KERNEL_H
