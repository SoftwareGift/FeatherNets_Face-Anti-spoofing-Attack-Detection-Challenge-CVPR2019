/*
 * gles_std.h - GLES std
 *
 *  Copyright (c) 2018 Intel Corporation
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

#ifndef XCAM_GLES_STD_H
#define XCAM_GLES_STD_H

#include <xcam_std.h>
#include <smartptr.h>
#include <GLES3/gl3.h>

#if HAVE_GLES_32
#include <GLES3/gl32.h>
#else
#include <GLES3/gl31.h>
#endif

#define XCAM_GL_NAME_LENGTH 64

#endif // XCAM_GLES_STD_H
