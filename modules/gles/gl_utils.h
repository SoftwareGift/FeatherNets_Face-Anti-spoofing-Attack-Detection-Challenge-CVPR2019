/*
 * gl_utils.h - GL utilities class
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

#ifndef XCAM_GL_UTILS_H
#define XCAM_GL_UTILS_H

#include <gles/gl_video_buffer.h>

namespace XCam {

SmartPtr<GLBuffer> get_glbuffer (const SmartPtr<VideoBuffer> &buf);

}

#endif // XCAM_GL_UTILS_H