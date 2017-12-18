/*
 * cl_intel_context.cpp - CL intel context
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

#include "cl_intel_context.h"
#include "cl_device.h"
#include "cl_va_memory.h"

#define OCL_EXT_NAME_CREATE_BUFFER_FROM_LIBVA_INTEL "clCreateBufferFromLibvaIntel"
#define OCL_EXT_NAME_CREATE_BUFFER_FROM_FD_INTEL    "clCreateBufferFromFdINTEL"
#define OCL_EXT_NAME_CREATE_IMAGE_FROM_LIBVA_INTEL  "clCreateImageFromLibvaIntel"
#define OCL_EXT_NAME_CREATE_IMAGE_FROM_FD_INTEL     "clCreateImageFromFdINTEL"
#define OCL_EXT_NAME_GET_MEM_OBJECT_FD_INTEL        "clGetMemObjectFdIntel"

namespace XCam {

CLIntelContext::CLIntelContext (SmartPtr<CLDevice> &device)
    : CLContext (device)
{
}

cl_mem
CLIntelContext::create_va_buffer (uint32_t bo_name)
{
    cl_mem mem_id = NULL;
    cl_int errcode = CL_SUCCESS;
    if (!is_valid())
        return NULL;

    clCreateBufferFromLibvaIntel_fn oclCreateBufferFromLibvaIntel =
        (clCreateBufferFromLibvaIntel_fn) _device->get_extension_function (OCL_EXT_NAME_CREATE_BUFFER_FROM_LIBVA_INTEL);
    XCAM_FAIL_RETURN(ERROR, oclCreateBufferFromLibvaIntel, NULL, "create buffer failed since extension was not found");

    mem_id = oclCreateBufferFromLibvaIntel (_context_id, bo_name, &errcode);
    XCAM_FAIL_RETURN(
        WARNING,
        errcode == CL_SUCCESS,
        NULL,
        "create cl memory from va image failed");
    return mem_id;
}

cl_mem
CLIntelContext::import_dma_buffer (const cl_import_buffer_info_intel &import_info)
{
    cl_mem mem_id = NULL;
    cl_int errcode = CL_SUCCESS;
    if (!is_valid())
        return NULL;

    clCreateBufferFromFdINTEL_fn oclCreateBufferFromFdINTEL =
        (clCreateBufferFromFdINTEL_fn) _device->get_extension_function (OCL_EXT_NAME_CREATE_BUFFER_FROM_FD_INTEL);
    XCAM_FAIL_RETURN(ERROR, oclCreateBufferFromFdINTEL, NULL, "import buffer failed since extension was not found");

    mem_id = oclCreateBufferFromFdINTEL (_context_id, &import_info, &errcode);
    XCAM_FAIL_RETURN(
        WARNING,
        errcode == CL_SUCCESS,
        NULL,
        "import cl memory from dma buffer failed");

    return mem_id;
}

cl_mem
CLIntelContext::create_va_image (const cl_libva_image &image_info)
{
    cl_mem mem_id = NULL;
    cl_int errcode = CL_SUCCESS;
    if (!is_valid())
        return NULL;

    clCreateImageFromLibvaIntel_fn oclCreateImageFromLibvaIntel =
        (clCreateImageFromLibvaIntel_fn) _device->get_extension_function (OCL_EXT_NAME_CREATE_IMAGE_FROM_LIBVA_INTEL);
    XCAM_FAIL_RETURN(ERROR, oclCreateImageFromLibvaIntel, NULL, "create image failed since extension was not found");

    mem_id = oclCreateImageFromLibvaIntel (_context_id, &image_info, &errcode);
    XCAM_FAIL_RETURN(
        WARNING,
        errcode == CL_SUCCESS,
        NULL,
        "create cl memory from va image failed");
    return mem_id;
}

cl_mem
CLIntelContext::import_dma_image (const cl_import_image_info_intel &import_info)
{
    cl_mem mem_id = NULL;
    cl_int errcode = CL_SUCCESS;
    if (!is_valid())
        return NULL;

    clCreateImageFromFdINTEL_fn oclCreateImageFromFdINTEL =
        (clCreateImageFromFdINTEL_fn) _device->get_extension_function (OCL_EXT_NAME_CREATE_IMAGE_FROM_FD_INTEL);
    XCAM_FAIL_RETURN(ERROR, oclCreateImageFromFdINTEL, NULL, "create image failed since extension was not found");

    mem_id = oclCreateImageFromFdINTEL (_context_id, &import_info, &errcode);
    XCAM_FAIL_RETURN(
        WARNING,
        errcode == CL_SUCCESS,
        NULL,
        "import cl memory from dma image failed, errcode:%d", errcode);

    return mem_id;
}

int32_t
CLIntelContext::export_mem_fd (cl_mem mem_id)
{
    cl_int errcode = CL_SUCCESS;
    int32_t fd = -1;

    clGetMemObjectFdIntel_fn oclGetMemObjectFdIntel =
        (clGetMemObjectFdIntel_fn) _device->get_extension_function (OCL_EXT_NAME_GET_MEM_OBJECT_FD_INTEL);
    XCAM_FAIL_RETURN(ERROR, oclGetMemObjectFdIntel, -1, "export fd failed since extension was not found");

    XCAM_ASSERT (mem_id);
    errcode = oclGetMemObjectFdIntel (_context_id, mem_id, &fd);
    XCAM_FAIL_RETURN (
        WARNING,
        errcode == CL_SUCCESS,
        -1,
        "export cl mem fd failed");
    return fd;
}

};
