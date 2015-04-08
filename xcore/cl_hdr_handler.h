/*
 * cl_hdr_handler.h - CL hdr handler
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

#ifndef XCAM_CL_HDR_HANLDER_H
#define XCAM_CL_HDR_HANLDER_H

#include "xcam_utils.h"
#include "cl_image_handler.h"

namespace XCam {

enum CLHdrType {
    CL_HDR_TYPE_RGB,
    CL_HDR_TYPE_LAB,
};

SmartPtr<CLImageHandler>
create_cl_hdr_image_handler (SmartPtr<CLContext> &context, CLHdrType type);

};

#endif //XCAM_CL_HDR_HANLDER_H
