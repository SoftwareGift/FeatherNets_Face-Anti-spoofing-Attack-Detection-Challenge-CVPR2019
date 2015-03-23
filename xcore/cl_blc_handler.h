/*
 * cl_blc_handler.h - CL black level correction handler
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

#ifndef XCAM_CL_BLC_HANLDER_H
#define XCAM_CL_BLC_HANLDER_H

#include "xcam_utils.h"
#include "cl_image_handler.h"

#define XCAM_CL_BLACK_LEVEL    0x3c
#define XCAM_CL_10BIT_NOR      0x400   /* Normalization for 10bit data */

namespace XCam {

class CLBlcImageKernel
    : public CLImageKernel
{

public:
    /*  Black level correction configuration
     *
    */
    typedef struct
    {
        cl_float  level_gr;  /* Black level for GR pixels */
        cl_float  level_r;   /* Black level for R pixels */
        cl_float  level_b;   /* Black level for B pixels */
        cl_float  level_gb;  /* Black level for GB pixels */
    } BLCConfig;

public:
    explicit CLBlcImageKernel (SmartPtr<CLContext> &context);

protected:
    virtual XCamReturn prepare_arguments (
        SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
        CLArgument args[], uint32_t &arg_count,
        CLWorkSize &work_size);
    BLCConfig _blc_config;

private:
    XCAM_DEAD_COPY (CLBlcImageKernel);
};

SmartPtr<CLImageHandler>
create_cl_blc_image_handler (SmartPtr<CLContext> &context);

};

#endif //XCAM_CL_BLC_HANLDER_H
