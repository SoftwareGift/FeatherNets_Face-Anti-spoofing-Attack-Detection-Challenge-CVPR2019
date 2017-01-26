/*
 * cl_dpc_handler.h - CL defect pixel correction handler
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

#ifndef XCAM_CL_DPC_HANLDER_H
#define XCAM_CL_DPC_HANLDER_H

#include "xcam_utils.h"
#include "cl_image_handler.h"
#include "base/xcam_3a_result.h"

namespace XCam {

#define XCAM_CL_DPC_DEFAULT_GAIN 1.0
#define XCAM_CL_DPC_DEFAULT_THRESHOLD 0.125

typedef struct {
    cl_float gain;   /* The sensitivity of mis-correction.     */
    cl_float  gr_threshold;  /* GR threshold of defect pixel correction */
    cl_float  r_threshold;   /* R threshold of defect pixel correction */
    cl_float  b_threshold;   /* B threshold of defect pixel correction */
    cl_float  gb_threshold;   /* GB  threshold of defect pixel correction */
} CLDPCConfig;

class CLDpcImageKernel
    : public CLImageKernel
{
public:
    explicit CLDpcImageKernel (SmartPtr<CLContext> &context);
    bool set_dpc (CLDPCConfig dpc);

protected:
    virtual XCamReturn prepare_arguments (
        SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
        CLArgument args[], uint32_t &arg_count,
        CLWorkSize &work_size);

private:
    XCAM_DEAD_COPY (CLDpcImageKernel);
    CLDPCConfig _dpc_config;
};

class CLDpcImageHandler
    : public CLImageHandler
{
public:
    explicit CLDpcImageHandler (const char *name);
    bool set_dpc_config (const XCam3aResultDefectPixel &dpc);
    bool set_dpc_kernel(SmartPtr<CLDpcImageKernel> &kernel);

private:
    XCAM_DEAD_COPY (CLDpcImageHandler);
    SmartPtr<CLDpcImageKernel> _dpc_kernel;
};

SmartPtr<CLImageHandler>
create_cl_dpc_image_handler (SmartPtr<CLContext> &context);

};

#endif //XCAM_CL_DPC_HANLDER_H
