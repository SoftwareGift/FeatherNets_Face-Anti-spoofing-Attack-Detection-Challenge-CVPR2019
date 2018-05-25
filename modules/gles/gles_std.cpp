/*
 * gles_std.cpp - GLES std
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
 * Author: Wind Yuan <feng.yuan@intel.com>
 */

#include "gles_std.h"

namespace XCam {

const char *
gl_error_string (GLenum flag)
{
    static char str[XCAM_GL_NAME_LENGTH] = {'\0'};

    switch (flag)
    {
    case GL_NO_ERROR:
        snprintf (str, sizeof (str), "0x%04x:%s", flag, "GL_NO_ERROR");
        break;
    case GL_INVALID_ENUM:
        snprintf (str, sizeof (str), "0x%04x:%s", flag, "GL_INVALID_ENUM");
        break;
    case GL_INVALID_VALUE:
        snprintf (str, sizeof (str), "0x%04x:%s", flag, "GL_INVALID_VALUE");
        break;
    case GL_INVALID_OPERATION:
        snprintf (str, sizeof (str), "0x%04x:%s", flag, "GL_INVALID_OPERATION");
        break;
    case GL_INVALID_FRAMEBUFFER_OPERATION:
        snprintf (str, sizeof (str), "0x%04x:%s", flag, "GL_INVALID_FRAMEBUFFER_OPERATION");
        break;
    case GL_OUT_OF_MEMORY:
        snprintf (str, sizeof (str), "0x%04x:%s", flag, "GL_OUT_OF_MEMORY");
        break;
    default:
        snprintf (str, sizeof (str), "unknown flag:0x%04x", flag);
        XCAM_LOG_ERROR ("%s", str);
    }

    return str;
}

}
