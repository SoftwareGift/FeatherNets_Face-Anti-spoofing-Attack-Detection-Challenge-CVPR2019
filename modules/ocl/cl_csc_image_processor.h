/*
 * cl_3a_image_processor.h - CL 3A image processor
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

#ifndef XCAM_CL_CSC_IMAGE_PROCESSOR_H
#define XCAM_CL_CSC_IMAGE_PROCESSOR_H

#include <xcam_std.h>
#include <stats_callback_interface.h>
#include <ocl/cl_image_processor.h>

namespace XCam {

class CLCscImageHandler;

class CLCscImageProcessor
    : public CLImageProcessor
{

public:
    explicit CLCscImageProcessor ();
    virtual ~CLCscImageProcessor ();

private:
    virtual XCamReturn create_handlers ();
    XCAM_DEAD_COPY (CLCscImageProcessor);

private:
    SmartPtr<CLCscImageHandler>        _csc;
};

};
#endif //XCAM_CL_CSC_IMAGE_PROCESSOR_H
