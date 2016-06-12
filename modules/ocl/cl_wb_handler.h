/*
 * cl_wb_handler.h - CL white balance handler
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

#ifndef XCAM_CL_WB_HANLDER_H
#define XCAM_CL_WB_HANLDER_H

#include "xcam_utils.h"
#include "cl_image_handler.h"
#include "base/xcam_3a_result.h"

namespace XCam {

typedef struct {
    float           r_gain;
    float           gr_gain;
    float           gb_gain;
    float           b_gain;
} CLWBConfig;

class CLWbImageKernel
    : public CLImageKernel
{
public:
    explicit CLWbImageKernel (SmartPtr<CLContext> &context);
    bool set_wb (CLWBConfig wb);

protected:
    virtual XCamReturn prepare_arguments (
        SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
        CLArgument args[], uint32_t &arg_count,
        CLWorkSize &work_size);

private:
    XCAM_DEAD_COPY (CLWbImageKernel);
    CLWBConfig _wb_config;
};

class CLWbImageHandler
    : public CLImageHandler
{
public:
    explicit CLWbImageHandler (const char *name);
    bool set_wb_config (const XCam3aResultWhiteBalance &wb);
    bool set_wb_kernel(SmartPtr<CLWbImageKernel> &kernel);

private:
    XCAM_DEAD_COPY (CLWbImageHandler);
    SmartPtr<CLWbImageKernel> _wb_kernel;
};

SmartPtr<CLImageHandler>
create_cl_wb_image_handler (SmartPtr<CLContext> &context);

};

#endif //XCAM_CL_WB_HANLDER_H
