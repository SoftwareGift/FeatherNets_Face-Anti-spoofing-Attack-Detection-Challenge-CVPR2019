/*
 * cl_gamma_handler.cpp - CL gamma handler
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
#include <xcam_std.h>
#include "cl_gamma_handler.h"

float default_gamma_table[XCAM_GAMMA_TABLE_SIZE] = {
    0.000000, 1.000000, 2.000000, 3.000000, 4.000000, 5.000000, 6.000000, 7.000000, 8.000000,
    9.000000, 10.000000, 11.000000, 12.000000, 13.000000, 14.000000, 15.000000, 16.000000,
    17.000000, 18.000000, 19.000000, 20.000000, 21.000000, 22.000000, 23.000000, 24.000000,
    25.000000, 26.000000, 27.000000, 28.000000, 29.000000, 30.000000, 31.000000, 32.000000,
    33.000000, 34.000000, 35.000000, 36.000000, 37.000000, 38.000000, 39.000000, 40.000000,
    41.000000, 42.000000, 43.000000, 44.000000, 45.000000, 46.000000, 47.000000, 48.000000,
    49.000000, 50.000000, 51.000000, 52.000000, 53.000000, 54.000000, 55.000000, 56.000000,
    57.000000, 58.000000, 59.000000, 60.000000, 61.000000, 62.000000, 63.000000, 64.000000,
    65.000000, 66.000000, 67.000000, 68.000000, 69.000000, 70.000000, 71.000000, 72.000000,
    73.000000, 74.000000, 75.000000, 76.000000, 77.000000, 78.000000, 79.000000, 80.000000,
    81.000000, 82.000000, 83.000000, 84.000000, 85.000000, 86.000000, 87.000000, 88.000000,
    89.000000, 90.000000, 91.000000, 92.000000, 93.000000, 94.000000, 95.000000, 96.000000,
    97.000000, 98.000000, 99.000000, 100.000000, 101.000000, 102.000000, 103.000000, 104.000000,
    105.000000, 106.000000, 107.000000, 108.000000, 109.000000, 110.000000, 111.000000, 112.000000,
    113.000000, 114.000000, 115.000000, 116.000000, 117.000000, 118.000000, 119.000000, 120.000000,
    121.000000, 122.000000, 123.000000, 124.000000, 125.000000, 126.000000, 127.000000, 128.000000,
    129.000000, 130.000000, 131.000000, 132.000000, 133.000000, 134.000000, 135.000000, 136.000000,
    137.000000, 138.000000, 139.000000, 140.000000, 141.000000, 142.000000, 143.000000, 144.000000,
    145.000000, 146.000000, 147.000000, 148.000000, 149.000000, 150.000000, 151.000000, 152.000000,
    153.000000, 154.000000, 155.000000, 156.000000, 157.000000, 158.000000, 159.000000, 160.000000,
    161.000000, 162.000000, 163.000000, 164.000000, 165.000000, 166.000000, 167.000000, 168.000000,
    169.000000, 170.000000, 171.000000, 172.000000, 173.000000, 174.000000, 175.000000, 176.000000,
    177.000000, 178.000000, 179.000000, 180.000000, 181.000000, 182.000000, 183.000000, 184.000000,
    185.000000, 186.000000, 187.000000, 188.000000, 189.000000, 190.000000, 191.000000, 192.000000,
    193.000000, 194.000000, 195.000000, 196.000000, 197.000000, 198.000000, 199.000000, 200.000000,
    201.000000, 202.000000, 203.000000, 204.000000, 205.000000, 206.000000, 207.000000, 208.000000,
    209.000000, 210.000000, 211.000000, 212.000000, 213.000000, 214.000000, 215.000000, 216.000000,
    217.000000, 218.000000, 219.000000, 220.000000, 221.000000, 222.000000, 223.000000, 224.000000,
    225.000000, 226.000000, 227.000000, 228.000000, 229.000000, 230.000000, 231.000000, 232.000000,
    233.000000, 234.000000, 235.000000, 236.000000, 237.000000, 238.000000, 239.000000, 240.000000,
    241.000000, 242.000000, 243.000000, 244.000000, 245.000000, 246.000000, 247.000000, 248.000000,
    249.000000, 250.000000, 251.000000, 252.000000, 253.000000, 254.000000, 255.000000
};

namespace XCam {

CLGammaImageKernel::CLGammaImageKernel (SmartPtr<CLContext> &context)
    : CLImageKernel (context, "kernel_gamma", false)
{
    set_gamma(default_gamma_table);
}

XCamReturn
CLGammaImageKernel::prepare_arguments (
    SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
    CLArgument args[], uint32_t &arg_count,
    CLWorkSize &work_size)
{
    SmartPtr<CLContext> context = get_context ();

    _image_in = new CLVaImage (context, input);
    _image_out = new CLVaImage (context, output);
    _gamma_table_buffer = new CLBuffer(
        context, sizeof(float)*XCAM_GAMMA_TABLE_SIZE,
        CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR , &_gamma_table);
    //CL_MEM_READ_ONLY

    XCAM_ASSERT (_image_in->is_valid () && _image_out->is_valid () && _gamma_table_buffer->is_valid());
    XCAM_FAIL_RETURN (
        WARNING,
        _image_in->is_valid () && _image_out->is_valid () && _gamma_table_buffer->is_valid(),
        XCAM_RETURN_ERROR_MEM,
        "cl image kernel(%s) in/out memory not available", get_kernel_name ());


    //set args;
    args[0].arg_adress = &_image_in->get_mem_id ();
    args[0].arg_size = sizeof (cl_mem);
    args[1].arg_adress = &_image_out->get_mem_id ();
    args[1].arg_size = sizeof (cl_mem);
    args[2].arg_adress = &_gamma_table_buffer->get_mem_id();
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
CLGammaImageKernel::set_gamma (float *gamma)
{
    memcpy(_gamma_table, gamma, sizeof(float)*XCAM_GAMMA_TABLE_SIZE);
    return true;
}

CLGammaImageHandler::CLGammaImageHandler (const char *name)
    : CLImageHandler (name),
      _brightness_impact (1)
{
}

bool
CLGammaImageHandler::set_gamma_table (const XCam3aResultGammaTable &gamma)
{
    float gamma_table[XCAM_GAMMA_TABLE_SIZE];

    for(int i = 0; i < XCAM_GAMMA_TABLE_SIZE; i++)
    {
        gamma_table[i] = (float)gamma.table[i] * _brightness_impact;
        if(gamma_table[i] > 255) gamma_table[i] = 255;
    }
    _gamma_kernel->set_gamma(gamma_table);
    return true;
}

bool
CLGammaImageHandler::set_manual_brightness (float level)
{
    _brightness_impact = level + 1;
    _brightness_impact = (_brightness_impact <= 0) ? 0 : _brightness_impact;
    return true;
}

bool
CLGammaImageHandler::set_gamma_kernel(SmartPtr<CLGammaImageKernel> &kernel)
{
    SmartPtr<CLImageKernel> image_kernel = kernel;
    add_kernel (image_kernel);
    _gamma_kernel = kernel;
    return true;
}

SmartPtr<CLImageHandler>
create_cl_gamma_image_handler (SmartPtr<CLContext> &context)
{
    SmartPtr<CLGammaImageHandler> gamma_handler;
    SmartPtr<CLGammaImageKernel> gamma_kernel;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    gamma_kernel = new CLGammaImageKernel (context);
    {
        XCAM_CL_KERNEL_FUNC_SOURCE_BEGIN(kernel_gamma)
#include "kernel_gamma.clx"
        XCAM_CL_KERNEL_FUNC_END;
        ret = gamma_kernel->load_from_source (kernel_gamma_body, strlen (kernel_gamma_body));
        XCAM_FAIL_RETURN (
            WARNING,
            ret == XCAM_RETURN_NO_ERROR,
            NULL,
            "CL image handler(%s) load source failed", gamma_kernel->get_kernel_name());
    }
    XCAM_ASSERT (gamma_kernel->is_valid ());
    gamma_handler = new CLGammaImageHandler ("cl_handler_gamma");
    gamma_handler->set_gamma_kernel (gamma_kernel);

    return gamma_handler;
}

}
