/*
 * cl_gauss_handler.h - CL gauss handler.
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

#ifndef XCAM_CL_GAUSS_HANLDER_H
#define XCAM_CL_GAUSS_HANLDER_H

#include "xcam_utils.h"
#include "cl_image_handler.h"
#include "base/xcam_3a_result.h"
#include "x3a_stats_pool.h"


#define XCAM_GAUSS_TABLE_SIZE 5

namespace XCam {

class CLGaussImageKernel
    : public CLImageKernel
{
public:
    explicit CLGaussImageKernel (SmartPtr<CLContext> &context);
    bool set_gaussian(int size, float sigma);

protected:
    virtual XCamReturn prepare_arguments (
        SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
        CLArgument args[], uint32_t &arg_count,
        CLWorkSize &work_size);

protected:
    uint32_t _vertical_offset_in;
    uint32_t _vertical_offset_out;
    SmartPtr<CLBuffer>  _g_table_buffer;
    float _g_table[XCAM_GAUSS_TABLE_SIZE*XCAM_GAUSS_TABLE_SIZE];
private:
    XCAM_DEAD_COPY (CLGaussImageKernel);
};

class CLGaussImageHandler
    : public CLImageHandler
{
public:
    explicit CLGaussImageHandler (const char *name);
    bool set_gauss_kernel(SmartPtr<CLGaussImageKernel> &kernel);
    bool set_gaussian_table(int size, float sigma);

private:
    XCAM_DEAD_COPY (CLGaussImageHandler);
    SmartPtr<CLGaussImageKernel> _gauss_kernel;
};

SmartPtr<CLImageHandler>
create_cl_gauss_image_handler (SmartPtr<CLContext> &context);

};

#endif //XCAM_CL_GAUSS_HANLDER_H
