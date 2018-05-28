/*
 * egl_utils.h - EGL utilities
 *
 *  Copyright (c) 2018 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Author: Yinhang Liu <yinhangx.liu@intel.com>
 */

#ifndef XCAM_EGL_UTILS_H
#define XCAM_EGL_UTILS_H

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <xcam_std.h>

namespace XCam {
namespace EGL {

inline EGLint get_error ()
{
    return eglGetError ();
}

const char *error_string (EGLint flag);

}
}
#endif // XCAM_EGL_UTILS_H
