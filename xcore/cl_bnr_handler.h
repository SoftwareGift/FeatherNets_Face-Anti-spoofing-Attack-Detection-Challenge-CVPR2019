/*
 * cl_bnr_handler.h - CL bnr handler
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
 * Author: Shincy Tu <shincy.tu@intel.com>
 */

#ifndef XCAM_CL_BNR_HANLDER_H
#define XCAM_CL_BNR_HANLDER_H

#include "xcam_utils.h"
#include "cl_image_handler.h"
#include "base/xcam_3a_result.h"
#include "cl_denoise_handler.h"


namespace XCam {

#define XCAM_CL_BNR_GAIN_DEFAULT         0.2
#define XCAM_CL_BNR_DIRECTION_DEFAULT    0.01

/*  Bayer noise reduction configuration  */
typedef struct  {
    cl_float  bnr_gain;  /* Strength of noise reduction */
    cl_float  direction;   /* Sensitivity of edge */
} CLBNRConfig;


class CLBnrImageKernel
    : public CLImageKernel
{
public:
    explicit CLBnrImageKernel (SmartPtr<CLContext> &context);
    bool set_bnr (CLBNRConfig bnr);

protected:
    virtual XCamReturn prepare_arguments (
        SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
        CLArgument args[], uint32_t &arg_count,
        CLWorkSize &work_size);

private:
    XCAM_DEAD_COPY (CLBnrImageKernel);
    CLBNRConfig _bnr_config;
};

class CLBnrImageHandler
    : public CLImageHandler
{
public:
    explicit CLBnrImageHandler (const char *name);
    bool set_bnr_config (const XCam3aResultBayerNoiseReduction &bnr);
    bool set_simple_kernel(SmartPtr<CLBnrImageKernel> &kernel);

private:
    XCAM_DEAD_COPY (CLBnrImageHandler);
    SmartPtr<CLBnrImageKernel> _simple_kernel;
};

SmartPtr<CLImageHandler>
create_cl_bnr_image_handler (SmartPtr<CLContext> &context);

};

#endif //XCAM_CL_BNR_HANLDER_H
