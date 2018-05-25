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
    , _need_configure (true)
    , _enable_allocator (true)
{
}

GLImageHandler::~GLImageHandler ()
{
}

bool
GLImageHandler::set_out_video_info (const VideoBufferInfo &info)
{
    XCAM_ASSERT (info.width && info.height && info.format);
    _out_video_info = info;

    return true;
}

bool
GLImageHandler::enable_allocator (bool enable)
{
    _enable_allocator = enable;
    return true;
}

XCamReturn
GLImageHandler::create_allocator ()
{
    XCAM_ASSERT (_need_configure);
    if (_enable_allocator) {
        XCAM_FAIL_RETURN (
            ERROR, _out_video_info.is_valid (), XCAM_RETURN_ERROR_PARAM,
            "GLImageHandler(%s) create allocator failed, out video info was not set",
            XCAM_STR (get_name ()));

        set_allocator (new GLVideoBufferPool);
        XCamReturn ret = reserve_buffers (_out_video_info, XCAM_GL_RESERVED_BUF_COUNT);
        XCAM_FAIL_RETURN (
            ERROR, ret == XCAM_RETURN_NO_ERROR, ret,
            "GLImageHandler(%s) reserve buffer failed", XCAM_STR (get_name ()));
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
GLImageHandler::execute_buffer (const SmartPtr<ImageHandler::Parameters> &param, bool sync)
{
    XCAM_UNUSED (sync);

    XCAM_FAIL_RETURN (
        ERROR, param.ptr (), XCAM_RETURN_ERROR_PARAM,
        "GLImageHandler(%s) parameters is null", XCAM_STR (get_name ()));

    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    if (_need_configure) {
        ret = configure_resource (param);
        XCAM_FAIL_RETURN (
            ERROR, xcam_ret_is_ok (ret), ret,
            "GLImageHandler(%s) configure resource failed", XCAM_STR (get_name ()));

        ret = create_allocator ();
        XCAM_FAIL_RETURN (
            ERROR, xcam_ret_is_ok (ret), ret,
            "GLImageHandler(%s) create allocator failed", XCAM_STR (get_name ()));

        _need_configure = false;
    }

    if (!param->out_buf.ptr () && _enable_allocator) {
        param->out_buf = get_free_buf ();
        XCAM_FAIL_RETURN (
            ERROR, param->out_buf.ptr (), XCAM_RETURN_ERROR_PARAM,
            "GLImageHandler(%s) get output buffer failed from allocator", XCAM_STR (get_name ()));
    }

    ret = start_work (param);
    if (!xcam_ret_is_ok (ret))
        XCAM_LOG_WARNING ("GLImageHandler(%s) start work failed", XCAM_STR (get_name ()));

    return ret;
}

void
GLImageHandler::execute_done (const SmartPtr<ImageHandler::Parameters> &param, XCamReturn err)
{
    XCAM_ASSERT (param.ptr ());

    if (err < XCAM_RETURN_NO_ERROR) {
        XCAM_LOG_WARNING (
            "GLImageHandler(%s) broken with errno(%d)", XCAM_STR (get_name ()), (int)err);
        return;
    }

    if (err > XCAM_RETURN_NO_ERROR) {
        XCAM_LOG_WARNING (
            "GLImageHandler(%s) continued with errno(%d)", XCAM_STR (get_name ()), (int)err);
    }

    execute_status_check (param, err);
}

}
