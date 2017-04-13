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

#define ENABLE_DEBUG_KERNEL 0

#define XCAM_CL_KERNEL_DEFAULT_WORK_DIM 2
#define XCAM_CL_KERNEL_DEFAULT_LOCAL_WORK_SIZE 0

namespace XCam {

typedef std::map<std::string, SmartPtr<CLKernel> > KernelMap;

CLKernel::CLKernel (SmartPtr<CLContext> &context, const char *name)
    : _name (NULL)
    , _kernel_id (NULL)
    , _context (context)
    , _work_dim (0)
{
    XCAM_ASSERT (context.ptr ());
    //XCAM_ASSERT (name);

    if (name)
        _name = strndup (name, XCAM_MAX_STR_SIZE);

    set_default_work_size ();

    XCAM_OBJ_PROFILING_INIT;
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
    if (!_parent_kernel.ptr ())
        _context->destroy_kernel_id (_kernel_id);
}

static void
get_string_key_id (const char *str, uint32_t len, uint8_t key_id[8])
{
    uint32_t key[2];
    uint32_t *ptr = (uint32_t*)(str);
    uint32_t aligned_len = 0;
    uint32_t i = 0;

    xcam_mem_clear (key);
    if (!len)
        len = strlen (str);
    aligned_len = XCAM_ALIGN_DOWN (len, 8);

    for (i = 0; i < aligned_len / 8; ++i) {
        key[0] ^= ptr[0];
        key[1] ^= ptr[1];
        ptr += 2;
    }
    memcpy (key_id, key, 8);
    len -= aligned_len;
    str += aligned_len;
    for (i = 0; i < len; ++i) {
        key_id[i] ^= (uint8_t)str[i];
    }
}

XCamReturn
CLKernel::build_kernel (const XCamKernelInfo& info, const char* options)
{
    static KernelMap kernel_map;
    static Mutex map_mutex;

    KernelMap::iterator i_kernel;
    SmartPtr<CLKernel> single_kernel;
    char key_str[1024];
    uint8_t body_key[8];
    std::string key;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    XCAM_FAIL_RETURN (ERROR, info.kernel_name, XCAM_RETURN_ERROR_PARAM, "build kernel failed since kernel name null");

    xcam_mem_clear (body_key);
    get_string_key_id (info.kernel_body, info.kernel_body_len, body_key);
    snprintf (
        key_str, sizeof(key_str),
        "%s#%02x%02x%02x%02x%02x%02x%02x%02x#%s",
        info.kernel_name,
        body_key[0], body_key[1], body_key[2], body_key[3], body_key[4], body_key[5], body_key[6], body_key[7],
        XCAM_STR(options));
    key = key_str;

    {
        SmartLock locker (map_mutex);
        i_kernel = kernel_map.find (key);
        if (i_kernel == kernel_map.end ()) {
            SmartPtr<CLContext>  context = get_context ();
            single_kernel = new CLKernel (context, info.kernel_name);
            XCAM_ASSERT (single_kernel.ptr ());
            ret = single_kernel->load_from_source (info.kernel_body, strlen (info.kernel_body), NULL, NULL, options);
            XCAM_FAIL_RETURN (
                ERROR, ret == XCAM_RETURN_NO_ERROR, ret,
                "build kernel(%s) from source failed", key_str);
            //kernel_map.insert (std::make_pair (key, single_kernel));
            kernel_map[key] = single_kernel;
        } else
            single_kernel = i_kernel->second;
    }

    XCAM_FAIL_RETURN (
        ERROR, (single_kernel.ptr () && single_kernel->is_valid ()), XCAM_RETURN_ERROR_UNKNOWN,
        "build kernel(%s) failed, unknown error", key_str);

    ret = this->clone (single_kernel);
    XCAM_FAIL_RETURN (
        ERROR, ret == XCAM_RETURN_NO_ERROR, ret,
        "load kernel(%s) from kernel failed", key_str);
    return ret;
}

XCamReturn
CLKernel::load_from_source (
    const char *source, size_t length,
    uint8_t **gen_binary, size_t *binary_size,
    const char *build_option)
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
            gen_binary, binary_size,
            build_option);
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
            CLContext::KERNEL_BUILD_BINARY,
            NULL, NULL,
            NULL);
    XCAM_FAIL_RETURN(
        WARNING,
        new_kernel_id != NULL,
        XCAM_RETURN_ERROR_CL,
        "cl kernel(%s) load from binary failed", XCAM_STR (_name));

    _kernel_id = new_kernel_id;
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CLKernel::clone (SmartPtr<CLKernel> kernel)
{
    XCAM_FAIL_RETURN (
        WARNING,
        kernel.ptr () && kernel->is_valid (),
        XCAM_RETURN_ERROR_CL,
        "cl kernel(%s) load from kernel failed", XCAM_STR (_name));
    _kernel_id = kernel->get_kernel_id ();
    _parent_kernel = kernel;
    if (!_name && kernel->get_kernel_name ()) {
        _name = strndup (kernel->get_kernel_name (), XCAM_MAX_STR_SIZE);
    }
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
            _name, i, (uint32_t)local [i], (uint32_t)dev_info.max_work_item_sizes [i]);
    }

    XCAM_FAIL_RETURN (
        WARNING,
        work_group_size == 0 || work_group_size <= dev_info.max_work_group_size,
        XCAM_RETURN_ERROR_PARAM,
        "kernel(%s) work-group-size:%d is greater than device max work-group-size(%d)",
        _name, work_group_size, (uint32_t)dev_info.max_work_group_size);

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
#if ENABLE_DEBUG_KERNEL
    XCAM_OBJ_PROFILING_START;
#endif

    XCamReturn ret = _context->execute_kernel (this, NULL, events, event_out);

#if ENABLE_DEBUG_KERNEL
    _context->finish ();
    char name[1024];
    snprintf (name, 1024, "%s-%p", XCAM_STR (_name), this);
    XCAM_OBJ_PROFILING_END (name, XCAM_OBJ_DUR_FRAME_NUM);
#endif
    return ret;
}

};
