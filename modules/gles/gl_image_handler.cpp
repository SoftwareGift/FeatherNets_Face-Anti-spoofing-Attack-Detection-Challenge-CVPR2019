/*
 * gl_image_handler.cpp - GL image handler implementation
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
 * Author: Yinhang Liu <yinhangx.liu@intel.com>
 */

#include "gl_image_handler.h"
#include "gl_video_buffer.h"

namespace XCam {

GLImageHandler::GLImageHandler (const char* name)
    : ImageHandler (name)
{
}

GLImageHandler::~GLImageHandler ()
{
}

SmartPtr<BufferPool>
GLImageHandler::create_allocator ()
{
    return new GLVideoBufferPool;
}

}
