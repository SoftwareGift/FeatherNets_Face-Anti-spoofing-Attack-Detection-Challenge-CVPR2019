/*
 * cl_gauss_handler.h - CL gauss handler.
 *
 *  Copyright (c) 2016 Intel Corporation
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

#ifndef XCAM_CL_GAUSS_HANLDER_H
#define XCAM_CL_GAUSS_HANLDER_H

#include <xcam_std.h>
#include <base/xcam_3a_result.h>
#include <x3a_stats_pool.h>
#include <ocl/cl_image_handler.h>

#define XCAM_GAUSS_DEFAULT_RADIUS 2
#define XCAM_GAUSS_DEFAULT_SIGMA 2.0f

namespace XCam {

class CLGaussImageKernel
    : public CLImageKernel
{
public:
    explicit CLGaussImageKernel (
        const SmartPtr<CLContext> &context, uint32_t radius, float sigma);
    virtual ~CLGaussImageKernel ();
    bool set_gaussian(uint32_t radius, float sigma);

protected:
    virtual XCamReturn prepare_arguments (CLArgList &args, CLWorkSize &work_size);

    // new virtual fucntions
    virtual SmartPtr<VideoBuffer> get_input_buf () = 0;
    virtual SmartPtr<VideoBuffer> get_output_buf () = 0;

protected:
    SmartPtr<CLBuffer>    _g_table_buffer;
    uint32_t              _g_radius;
    float                *_g_table;
};

class CLGaussImageHandler
    : public CLImageHandler
{
public:
    explicit CLGaussImageHandler (const SmartPtr<CLContext> &context, const char *name);
    bool set_gauss_kernel(SmartPtr<CLGaussImageKernel> &kernel);
    bool set_gaussian_table(int size, float sigma);

private:
    SmartPtr<CLGaussImageKernel> _gauss_kernel;
};

SmartPtr<CLImageHandler>
create_cl_gauss_image_handler (
    const SmartPtr<CLContext> &context,
    uint32_t radius = XCAM_GAUSS_DEFAULT_RADIUS,
    float sigma = XCAM_GAUSS_DEFAULT_SIGMA);

};

#endif //XCAM_CL_GAUSS_HANLDER_H
