/*
 * egl_utils.cpp - EGL utilities
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

#include "egl_utils.h"

namespace XCam {
namespace EGL {

const char *
error_string (EGLint flag)
{
    static char str[64] = {'\0'};

    switch (flag)
    {
    case EGL_SUCCESS:
        snprintf (str, sizeof (str), "0x%04x:%s", flag, "EGL_SUCCESS");
        break;
    case EGL_NOT_INITIALIZED:
        snprintf (str, sizeof (str), "0x%04x:%s", flag, "EGL_NOT_INITIALIZED");
        break;
    case EGL_BAD_ACCESS:
        snprintf (str, sizeof (str), "0x%04x:%s", flag, "EGL_BAD_ACCESS");
        break;
    case EGL_BAD_ALLOC:
        snprintf (str, sizeof (str), "0x%04x:%s", flag, "EGL_BAD_ALLOC");
        break;
    case EGL_BAD_ATTRIBUTE:
        snprintf (str, sizeof (str), "0x%04x:%s", flag, "EGL_BAD_ATTRIBUTE");
        break;
    case EGL_BAD_CONTEXT:
        snprintf (str, sizeof (str), "0x%04x:%s", flag, "EGL_BAD_CONTEXT");
        break;
    case EGL_BAD_CONFIG:
        snprintf (str, sizeof (str), "0x%04x:%s", flag, "EGL_BAD_CONFIG");
        break;
    case EGL_BAD_CURRENT_SURFACE:
        snprintf (str, sizeof (str), "0x%04x:%s", flag, "EGL_BAD_CURRENT_SURFACE");
        break;
    case EGL_BAD_DISPLAY:
        snprintf (str, sizeof (str), "0x%04x:%s", flag, "EGL_BAD_DISPLAY");
        break;
    case EGL_BAD_SURFACE:
        snprintf (str, sizeof (str), "0x%04x:%s", flag, "EGL_BAD_SURFACE");
        break;
    case EGL_BAD_MATCH:
        snprintf (str, sizeof (str), "0x%04x:%s", flag, "EGL_BAD_MATCH");
        break;
    case EGL_BAD_PARAMETER:
        snprintf (str, sizeof (str), "0x%04x:%s", flag, "EGL_BAD_PARAMETER");
        break;
    case EGL_BAD_NATIVE_PIXMAP:
        snprintf (str, sizeof (str), "0x%04x:%s", flag, "EGL_BAD_NATIVE_PIXMAP");
        break;
    case EGL_BAD_NATIVE_WINDOW:
        snprintf (str, sizeof (str), "0x%04x:%s", flag, "EGL_BAD_NATIVE_WINDOW");
        break;
    case EGL_CONTEXT_LOST:
        snprintf (str, sizeof (str), "0x%04x:%s", flag, "EGL_CONTEXT_LOST");
        break;
    default:
        snprintf (str, sizeof (str), "unknown flag:0x%04x", flag);
        XCAM_LOG_ERROR ("%s", str);
    }

    return str;
}

}
}
