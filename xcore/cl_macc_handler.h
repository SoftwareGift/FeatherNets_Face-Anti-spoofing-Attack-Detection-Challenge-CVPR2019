/*
 * cl_macc_handler.h - CL macc handler
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

#ifndef XCAM_CL_MACC_HANLDER_H
#define XCAM_CL_MACC_HANLDER_H

#include "xcam_utils.h"
#include "cl_image_handler.h"
#include "base/xcam_3a_result.h"

namespace XCam {

class CLMaccImageKernel
    : public CLImageKernel
{
public:
    explicit CLMaccImageKernel (SmartPtr<CLContext> &context);
    bool set_macc (float *macc);

protected:
    virtual XCamReturn prepare_arguments (
        SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
        CLArgument args[], uint32_t &arg_count,
        CLWorkSize &work_size);

private:
    XCAM_DEAD_COPY (CLMaccImageKernel);

    float               _macc_table[XCAM_CHROMA_AXIS_SIZE * XCAM_CHROMA_MATRIX_SIZE];
    SmartPtr<CLBuffer>  _macc_table_buffer;
};

class CLMaccImageHandler
    : public CLImageHandler
{
public:
    explicit CLMaccImageHandler (const char *name);
    bool set_macc_table (XCam3aResultMaccMatrix macc);
    bool set_macc_kernel(SmartPtr<CLMaccImageKernel> &kernel);

private:
    XCAM_DEAD_COPY (CLMaccImageHandler);

    SmartPtr<CLMaccImageKernel> _macc_kernel;
};

SmartPtr<CLImageHandler>
create_cl_macc_image_handler (SmartPtr<CLContext> &context);

};

#endif //XCAM_CL_Macc_HANLDER_H
