/*
 * cl_gamma_handler.h - CL gamma handler
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

#ifndef XCAM_CL_GAMMA_HANLDER_H
#define XCAM_CL_GAMMA_HANLDER_H

#include "xcam_utils.h"
#include "ocl/cl_image_handler.h"
#include "base/xcam_3a_result.h"

namespace XCam {

class CLGammaImageKernel
    : public CLImageKernel
{
public:
    explicit CLGammaImageKernel (SmartPtr<CLContext> &context);
    bool set_gamma (float *gamma);


protected:
    virtual XCamReturn prepare_arguments (
        SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
        CLArgument args[], uint32_t &arg_count,
        CLWorkSize &work_size);

private:
    XCAM_DEAD_COPY (CLGammaImageKernel);

    float               _gamma_table[XCAM_GAMMA_TABLE_SIZE];
    SmartPtr<CLBuffer>  _gamma_table_buffer;
};

class CLGammaImageHandler
    : public CLImageHandler
{
public:
    explicit CLGammaImageHandler (const char *name);
    bool set_gamma_table (const XCam3aResultGammaTable &gamma);
    bool set_gamma_kernel(SmartPtr<CLGammaImageKernel> &kernel);
    bool set_manual_brightness (float level);

private:
    XCAM_DEAD_COPY (CLGammaImageHandler);

    SmartPtr<CLGammaImageKernel> _gamma_kernel;
    float _brightness_impact;
};

SmartPtr<CLImageHandler>
create_cl_gamma_image_handler (SmartPtr<CLContext> &context);

};

#endif //XCAM_CL_GAMMA_HANLDER_H
