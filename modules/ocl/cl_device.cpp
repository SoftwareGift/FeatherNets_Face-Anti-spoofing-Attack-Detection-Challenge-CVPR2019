/*
 * cl_device.cpp - CL device
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

#include "cl_device.h"
#include "cl_context.h"
#if HAVE_LIBDRM
#include "intel/cl_intel_context.h"
#endif

namespace XCam {

SmartPtr<CLDevice> CLDevice::_instance;
Mutex CLDevice::_instance_mutex;

SmartPtr<CLDevice>
CLDevice::instance ()
{
    SmartLock locker(_instance_mutex);
    if (_instance.ptr())
        return _instance;

    _instance = new CLDevice ();
    // create default context
    if (_instance->is_inited() &&
            !_instance->create_default_context ()) {
        XCAM_LOG_WARNING ("CL device create default context failed");
    }

    return _instance;
}

CLDevice::CLDevice()
    : _platform_id (NULL)
    , _device_id (NULL)
    , _inited (false)
{
    if (!init()) {
        XCAM_LOG_WARNING ("CL device init failed");
    }
    XCAM_LOG_DEBUG ("CL device constructed");
}

CLDevice::~CLDevice ()
{
    XCAM_LOG_DEBUG ("CL device destructed");
}

SmartPtr<CLContext>
CLDevice::get_context ()
{
    //created in CLDevice construction
    return _default_context;
}

void *
CLDevice::get_extension_function (const char *func_name)
{
    XCAM_ASSERT (func_name);
    void *ext_func = NULL;

#if defined (CL_VERSION_1_2) && (CL_VERSION_1_2 == 1)
    ext_func = (void *) clGetExtensionFunctionAddressForPlatform (_platform_id, func_name);
#else
    ext_func = (void *) clGetExtensionFunctionAddress (func_name);
#endif
    if (!ext_func)
        XCAM_LOG_ERROR ("ocl driver get extension function (%s) failed", func_name);

    return ext_func;
}

void
CLDevice::terminate ()
{
    if (_default_context.ptr ()) {
        _default_context->terminate ();
        _default_context.release ();
    }
}

bool
CLDevice::init ()
{
    cl_platform_id platform_id = NULL;
    cl_device_id   device_id = NULL;
    cl_uint num_platform = 0;
    cl_uint num_device = 0;
    CLDevieInfo device_info;

    if (clGetPlatformIDs (1, &platform_id, &num_platform) != CL_SUCCESS)
    {
        XCAM_LOG_WARNING ("get cl platform ID failed");
        return false;
    }
    XCAM_ASSERT (num_platform >= 1);

    if (clGetDeviceIDs (platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &num_device) != CL_SUCCESS)
    {
        XCAM_LOG_WARNING ("get cl device ID failed");
        return false;
    }
    XCAM_ASSERT (num_device >= 1);

    // only query first device info
    if (!query_device_info (device_id, device_info)) {
        //continue
        XCAM_LOG_WARNING ("cl get device info failed but continue");
    } else {
        XCAM_LOG_INFO (
            "cl get device info,\n"
            "\tmax_compute_unit:%" PRIu32
            "\tmax_work_item_dims:%" PRIu32
            "\tmax_work_item_sizes:{%" PRIuS ", %" PRIuS ", %" PRIuS "}"
            "\tmax_work_group_size:%" PRIuS
            "\timage_pitch_alignment:%" PRIu32,
            device_info.max_compute_unit,
            device_info.max_work_item_dims,
            device_info.max_work_item_sizes[0], device_info.max_work_item_sizes[1], device_info.max_work_item_sizes[2],
            device_info.max_work_group_size,
            device_info.image_pitch_alignment);
    }

    // get platform name string length
    size_t sz = 0;
    if (clGetPlatformInfo(platform_id, CL_PLATFORM_NAME, 0, 0, &sz) != CL_SUCCESS)
    {
        XCAM_LOG_WARNING ("get cl platform name failed");
        return false;
    }

    // get platform name string
    if (sz >= XCAM_CL_MAX_STR_SIZE) {
        sz = XCAM_CL_MAX_STR_SIZE - 1;
    }
    if (clGetPlatformInfo(platform_id, CL_PLATFORM_NAME, sz, _platform_name, 0) != CL_SUCCESS)
    {
        XCAM_LOG_WARNING ("get cl platform name failed");
        return false;
    }

    _platform_id = platform_id;
    _device_id = device_id;
    _device_info = device_info;
    _platform_name[sz] = 0;
    _inited = true;
    return true;
}

bool
CLDevice::query_device_info (cl_device_id device_id, CLDevieInfo &info)
{
#undef XCAM_CL_GET_DEVICE_INFO
#define XCAM_CL_GET_DEVICE_INFO(name, val)                            \
    do {                                                              \
    if (clGetDeviceInfo (device_id, name, sizeof (val), &(val), NULL) != CL_SUCCESS) {   \
        XCAM_LOG_WARNING ("cl get device info(%s) failed", #name);    \
    } } while (0)

    XCAM_CL_GET_DEVICE_INFO (CL_DEVICE_MAX_COMPUTE_UNITS, info.max_compute_unit);
    XCAM_CL_GET_DEVICE_INFO (CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, info.max_work_item_dims);
    XCAM_CL_GET_DEVICE_INFO (CL_DEVICE_MAX_WORK_ITEM_SIZES, info.max_work_item_sizes);
    XCAM_CL_GET_DEVICE_INFO (CL_DEVICE_MAX_WORK_GROUP_SIZE, info.max_work_group_size);
    XCAM_CL_GET_DEVICE_INFO (CL_DEVICE_MAX_WORK_GROUP_SIZE, info.max_work_group_size);

    cl_uint alignment = 0;
    XCAM_CL_GET_DEVICE_INFO (CL_DEVICE_IMAGE_PITCH_ALIGNMENT, alignment);
    if (alignment)
        info.image_pitch_alignment = alignment;
    else
        info.image_pitch_alignment = 4;
    return true;
}

bool
CLDevice::create_default_context ()
{
    SmartPtr<CLContext> context;

#if HAVE_LIBDRM
    context = new CLIntelContext (_instance);
#else
    context = new CLContext (_instance);
#endif
    if (!context->is_valid())
        return false;

    // init first cmdqueue
    if (context->is_valid () && !context->init_cmd_queue (context)) {
        XCAM_LOG_ERROR ("CL context init cmd queue failed");
    }
    _default_context = context;
    return true;
}

};
