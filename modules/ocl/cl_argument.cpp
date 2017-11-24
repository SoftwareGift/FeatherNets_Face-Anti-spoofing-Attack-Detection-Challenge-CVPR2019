/*
 * cl_argument.cpp - CL kernel Argument
 *
 *  Copyright (c) 2017 Intel Corporation
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
 * Author: Wind Yuan <feng.yuan@intel.com>
 */

#include "cl_argument.h"

namespace XCam {


CLWorkSize::CLWorkSize ()
    : dim (XCAM_DEFAULT_IMAGE_DIM)
{
    xcam_mem_clear (global);
    xcam_mem_clear (local);
}

CLArgument::CLArgument (uint32_t size)
    : _arg_adress (NULL)
    , _arg_size (size)
{
}

CLArgument::~CLArgument ()
{
}

void
CLArgument::get_value (void *&adress, uint32_t &size)
{
    adress = _arg_adress;
    size = _arg_size;
}


}
