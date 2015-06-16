/*
 * cl_ee_handler.h - CL edge enhancement handler.
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

#ifndef XCAM_CL_EE_HANLDER_H
#define XCAM_CL_EE_HANLDER_H

#include "xcam_utils.h"
#include "cl_image_handler.h"
#include "base/xcam_3a_result.h"

namespace XCam {

typedef struct {
    float           ee_gain;
    float           ee_threshold;
    float           nr_gain;
} CLEeConfig;

class CLEeImageKernel
    : public CLImageKernel
{
public:
    explicit CLEeImageKernel (SmartPtr<CLContext> &context);
    bool set_ee_ee (const XCam3aResultEdgeEnhancement &ee);
    bool set_ee_nr (const XCam3aResultNoiseReduction &nr);

protected:
    virtual XCamReturn prepare_arguments (
        SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
        CLArgument args[], uint32_t &arg_count,
        CLWorkSize &work_size);

private:
    XCAM_DEAD_COPY (CLEeImageKernel);
    uint32_t _vertical_offset;
    CLEeConfig _ee_config;
};

class CLEeImageHandler
    : public CLImageHandler
{
public:
    explicit CLEeImageHandler (const char *name);
    bool set_ee_config_ee (const XCam3aResultEdgeEnhancement &ee);
    bool set_ee_config_nr (const XCam3aResultNoiseReduction &nr);
    bool set_ee_kernel(SmartPtr<CLEeImageKernel> &kernel);

private:
    XCAM_DEAD_COPY (CLEeImageHandler);
    SmartPtr<CLEeImageKernel> _ee_kernel;
};

SmartPtr<CLImageHandler>
create_cl_ee_image_handler (SmartPtr<CLContext> &context);

};

#endif //XCAM_CL_EE_HANLDER_H
