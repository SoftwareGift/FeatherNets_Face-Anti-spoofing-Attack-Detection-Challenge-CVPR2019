/*
 * cl_macc_handler.cpp - CL gamma handler
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
 * Author: wangfei <feix.w.wang@intel.com>
 */
#include "xcam_utils.h"
#include "cl_macc_handler.h"

float default_macc_table[XCAM_CHROMA_AXIS_SIZE*XCAM_CHROMA_MATRIX_SIZE] = {
    1.000000, 0.000000, 0.000000, 1.000000, 1.000000, 0.000000, 0.000000, 1.000000,
    1.000000, 0.000000, 0.000000, 1.000000, 1.000000, 0.000000, 0.000000, 1.000000,
    1.000000, 0.000000, 0.000000, 1.000000, 1.000000, 0.000000, 0.000000, 1.000000,
    1.000000, 0.000000, 0.000000, 1.000000, 1.000000, 0.000000, 0.000000, 1.000000,
    1.000000, 0.000000, 0.000000, 1.000000, 1.000000, 0.000000, 0.000000, 1.000000,
    1.000000, 0.000000, 0.000000, 1.000000, 1.000000, 0.000000, 0.000000, 1.000000,
    1.000000, 0.000000, 0.000000, 1.000000, 1.000000, 0.000000, 0.000000, 1.000000,
    1.000000, 0.000000, 0.000000, 1.000000, 1.000000, 0.000000, 0.000000, 1.000000
};

namespace XCam {

CLMaccImageKernel::CLMaccImageKernel (SmartPtr<CLContext> &context)
    : CLImageKernel (context, "kernel_macc", false)
{
    set_macc (default_macc_table);
}

XCamReturn
CLMaccImageKernel::prepare_arguments (
    SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
    CLArgument args[], uint32_t &arg_count,
    CLWorkSize &work_size)
{
    SmartPtr<CLContext> context = get_context ();

    _image_in = new CLVaImage (context, input);
    _image_out = new CLVaImage (context, output);
    _macc_table_buffer = new CLBuffer(
        context, sizeof(float)*XCAM_CHROMA_AXIS_SIZE * XCAM_CHROMA_MATRIX_SIZE,
        CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR , &_macc_table);

    XCAM_ASSERT (_image_in->is_valid () && _image_out->is_valid () && _macc_table_buffer->is_valid());
    XCAM_FAIL_RETURN (
        WARNING,
        _image_in->is_valid () && _image_out->is_valid () && _macc_table_buffer->is_valid(),
        XCAM_RETURN_ERROR_MEM,
        "cl image kernel(%s) in/out memory not available", get_kernel_name ());


    //set args;
    args[0].arg_adress = &_image_in->get_mem_id ();
    args[0].arg_size = sizeof (cl_mem);
    args[1].arg_adress = &_image_out->get_mem_id ();
    args[1].arg_size = sizeof (cl_mem);
    args[2].arg_adress = &_macc_table_buffer->get_mem_id();
    args[2].arg_size = sizeof (cl_mem);
    arg_count = 3;

    const CLImageDesc out_info = _image_out->get_image_desc ();
    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    work_size.global[0] = out_info.width / 4;
    work_size.global[1] = out_info.height / 2;
    work_size.local[0] = 4;
    work_size.local[1] = 4;

    return XCAM_RETURN_NO_ERROR;
}

bool
CLMaccImageKernel::set_macc (float *macc)
{
    memcpy(_macc_table, macc, sizeof(float)*XCAM_CHROMA_AXIS_SIZE * XCAM_CHROMA_MATRIX_SIZE);
    return true;
}
CLMaccImageHandler::CLMaccImageHandler (const char *name)
    : CLImageHandler (name)
{
}

bool
CLMaccImageHandler::set_macc_table (const XCam3aResultMaccMatrix &macc)
{
    float macc_table[XCAM_CHROMA_AXIS_SIZE * XCAM_CHROMA_MATRIX_SIZE];
    for(int i = 0; i < XCAM_CHROMA_AXIS_SIZE * XCAM_CHROMA_MATRIX_SIZE; i++)
        macc_table[i] = (float)macc.table[i];
    _macc_kernel->set_macc(macc_table);
    return true;
}

bool
CLMaccImageHandler::set_macc_kernel(SmartPtr<CLMaccImageKernel> &kernel)
{
    SmartPtr<CLImageKernel> image_kernel = kernel;
    add_kernel (image_kernel);
    _macc_kernel = kernel;
    return true;
}


SmartPtr<CLImageHandler>
create_cl_macc_image_handler (SmartPtr<CLContext> &context)
{
    SmartPtr<CLMaccImageHandler> macc_handler;
    SmartPtr<CLMaccImageKernel> macc_kernel;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    macc_kernel = new CLMaccImageKernel (context);
    {
        XCAM_CL_KERNEL_FUNC_SOURCE_BEGIN(kernel_macc)
#include "kernel_macc.clx"
        XCAM_CL_KERNEL_FUNC_END;
        ret = macc_kernel->load_from_source (kernel_macc_body, strlen (kernel_macc_body));
        XCAM_FAIL_RETURN (
            WARNING,
            ret == XCAM_RETURN_NO_ERROR,
            NULL,
            "CL image handler(%s) load source failed", macc_kernel->get_kernel_name());
    }
    XCAM_ASSERT (macc_kernel->is_valid ());
    macc_handler = new CLMaccImageHandler ("cl_handler_macc");
    macc_handler->set_macc_kernel (macc_kernel);

    return macc_handler;
}

}
