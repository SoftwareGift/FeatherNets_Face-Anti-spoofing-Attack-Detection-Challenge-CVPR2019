/*
 * cl_kernel.cpp - CL kernel
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

#include "cl_kernel.h"
#include "cl_context.h"
#include "cl_device.h"

#define XCAM_CL_KERNEL_DEFAULT_WORK_DIM 2
#define XCAM_CL_KERNEL_DEFAULT_LOCAL_WORK_SIZE 0

namespace XCam {

CLKernel::CLKernel(SmartPtr<CLContext> &context, const char *name)
    : _name (NULL)
    , _kernel_id (NULL)
    , _context (context)
    , _work_dim (0)
{
    XCAM_ASSERT (context.ptr ());
    XCAM_ASSERT (name);

    if (name)
        _name = strdup (name);

    set_default_work_size ();
}

CLKernel::~CLKernel ()
{
    destroy ();
    if (_name)
        xcam_free (_name);
}

void
CLKernel::destroy ()
{
    _context->destroy_kernel_id (_kernel_id);
}

XCamReturn
CLKernel::load_from_source (
    const char *source, size_t length,
    uint8_t **program_binaries, size_t *binary_sizes)
{
    cl_kernel new_kernel_id = NULL;

    XCAM_ASSERT (source);
    if (!source) {
        XCAM_LOG_WARNING ("kernel:%s source empty", XCAM_STR (_name));
        return XCAM_RETURN_ERROR_PARAM;
    }

    if (_kernel_id) {
        XCAM_LOG_WARNING ("kernel:%s already build yet", XCAM_STR (_name));
        return XCAM_RETURN_ERROR_PARAM;
    }

    XCAM_ASSERT (_context.ptr ());

    if (length == 0)
        length = strlen (source);

    new_kernel_id =
        _context->generate_kernel_id (
            this,
            (const uint8_t *)source, length,
            CLContext::KERNEL_BUILD_SOURCE,
            program_binaries, binary_sizes);
    XCAM_FAIL_RETURN(
        WARNING,
        new_kernel_id != NULL,
        XCAM_RETURN_ERROR_CL,
        "cl kernel(%s) load from source failed", XCAM_STR (_name));

    _kernel_id = new_kernel_id;
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CLKernel::load_from_binary (const uint8_t *binary, size_t length)
{
    cl_kernel new_kernel_id = NULL;

    XCAM_ASSERT (binary);
    if (!binary || !length) {
        XCAM_LOG_WARNING ("kernel:%s binary empty", XCAM_STR (_name));
        return XCAM_RETURN_ERROR_PARAM;
    }

    if (_kernel_id) {
        XCAM_LOG_WARNING ("kernel:%s already build yet", XCAM_STR (_name));
        return XCAM_RETURN_ERROR_PARAM;
    }

    XCAM_ASSERT (_context.ptr ());

    new_kernel_id =
        _context->generate_kernel_id (
            this,
            binary, length,
            CLContext::KERNEL_BUILD_BINARY);
    XCAM_FAIL_RETURN(
        WARNING,
        new_kernel_id != NULL,
        XCAM_RETURN_ERROR_CL,
        "cl kernel(%s) load from binary failed", XCAM_STR (_name));

    _kernel_id = new_kernel_id;
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CLKernel::set_argument (uint32_t arg_i, void *arg_addr, uint32_t arg_size)
{
    cl_int error_code = clSetKernelArg (_kernel_id, arg_i, arg_size, arg_addr);
    if (error_code != CL_SUCCESS) {
        XCAM_LOG_DEBUG ("kernel(%s) set arg_i(%d) failed", _name, arg_i);
        return XCAM_RETURN_ERROR_CL;
    }
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CLKernel::set_work_size (uint32_t dim, size_t *global, size_t *local)
{
    uint32_t i = 0;
    uint32_t work_group_size = 1;
    const CLDevieInfo &dev_info = CLDevice::instance ()->get_device_info ();

    XCAM_FAIL_RETURN (
        WARNING,
        dim <= dev_info.max_work_item_dims,
        XCAM_RETURN_ERROR_PARAM,
        "kernel(%s) work dims(%d) greater than device max dims(%d)",
        _name, dim, dev_info.max_work_item_dims);

    for (i = 0; i < dim; ++i) {
        work_group_size *= local [i];

        XCAM_FAIL_RETURN (
            WARNING,
            local [i] <= dev_info.max_work_item_sizes [i],
            XCAM_RETURN_ERROR_PARAM,
            "kernel(%s) work item(%d) size:%d is greater than device max work item size(%d)",
            _name, i, local [i], dev_info.max_work_item_sizes [i]);
    }

    XCAM_FAIL_RETURN (
        WARNING,
        work_group_size == 0 || work_group_size <= dev_info.max_work_group_size,
        XCAM_RETURN_ERROR_PARAM,
        "kernel(%s) work-group-size:%d is greater than device max work-group-size(%d)",
        _name, work_group_size, dev_info.max_work_group_size);

    _work_dim = dim;
    for (i = 0; i < dim; ++i) {
        _global_work_size [i] = global [i];
        _local_work_size [i] = local [i];
    }

    return XCAM_RETURN_NO_ERROR;
}

void
CLKernel::set_default_work_size ()
{
    _work_dim = XCAM_CL_KERNEL_DEFAULT_WORK_DIM;
    for (uint32_t i = 0; i < _work_dim; ++i) {
        //_global_work_size [i] = XCAM_CL_KERNEL_DEFAULT_GLOBAL_WORK_SIZE;
        _local_work_size [i] = XCAM_CL_KERNEL_DEFAULT_LOCAL_WORK_SIZE;
    }
}

XCamReturn
CLKernel::execute (
    CLEventList &events,
    SmartPtr<CLEvent> &event_out)
{
    XCAM_ASSERT (_context.ptr ());
    return _context->execute_kernel (this, NULL, events, event_out);
}

};
