/*
 * image_handler.cpp - image handler implementation
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

#include "image_handler.h"

namespace XCam {

ImageHandler::ImageHandler (const char* name)
    : _need_configure (true)
    , _enable_allocator (false)
    , _buf_capacity (0)
    , _name (NULL)
{
    if (name)
        _name = strndup (name, XCAM_MAX_STR_SIZE);
}

ImageHandler::~ImageHandler()
{
    xcam_mem_clear (_name);
}

bool
ImageHandler::set_out_video_info (const VideoBufferInfo &info)
{
    XCAM_ASSERT (info.width && info.height && info.format);
    _out_video_info = info;
    return true;
}

bool
ImageHandler::enable_allocator (bool enable, uint32_t buf_count)
{

    if (enable && !buf_count) {
        XCAM_LOG_ERROR (
            "ImageHandler(%s) enable allocator must with buf_count>0", XCAM_STR(get_name ()));
        return false;
    }

    _enable_allocator = enable;
    if (enable)
        _buf_capacity = buf_count;

    return true;
}

bool
ImageHandler::set_allocator (const SmartPtr<BufferPool> &allocator)
{
    XCAM_FAIL_RETURN (
        ERROR, allocator.ptr (), false,
        "ImageHandler(%s) set allocator(is NULL)", XCAM_STR(get_name ()));
    _allocator = allocator;
    return true;
}

XCamReturn
ImageHandler::configure_rest ()
{
    if (_enable_allocator) {
        XCAM_FAIL_RETURN (
            ERROR, _out_video_info.is_valid (), XCAM_RETURN_ERROR_PARAM,
            "image_hander(%s) configure reset failed before reserver buffer since out_video_info was not set",
            XCAM_STR (get_name ()));

        SmartPtr<BufferPool> allocator = create_allocator ();
        XCAM_FAIL_RETURN (
            ERROR, allocator.ptr (), XCAM_RETURN_ERROR_PARAM,
            "image_hander(%s) configure reset failed since allocator not created", XCAM_STR (get_name ()));
        _allocator = allocator;
        XCamReturn ret = reserve_buffers (_out_video_info, _buf_capacity);
        XCAM_FAIL_RETURN (
            ERROR, xcam_ret_is_ok (ret), ret,
            "soft_hander(%s) configure resource failed in reserving buffers", XCAM_STR (get_name ()));
    }
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
ImageHandler::execute_buffer (const SmartPtr<ImageHandler::Parameters> &param, bool sync)
{
    XCAM_UNUSED (sync);

    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    XCAM_FAIL_RETURN (
        ERROR, param.ptr (), XCAM_RETURN_ERROR_PARAM,
        "image_handler(%s) execute buffer failed, params is null",
        XCAM_STR (get_name ()));

    if (_need_configure) {
        ret = configure_resource (param);
        XCAM_FAIL_RETURN (
            WARNING, xcam_ret_is_ok (ret), ret,
            "image_handler(%s) configure resource failed", XCAM_STR (get_name ()));

        ret = configure_rest ();
        XCAM_FAIL_RETURN (
            WARNING, xcam_ret_is_ok (ret), ret,
            "image_handler(%s) configure rest failed", XCAM_STR (get_name ()));
        _need_configure = false;
    }

    if (!param->out_buf.ptr () && _enable_allocator) {
        param->out_buf = get_free_buf ();
        XCAM_FAIL_RETURN (
            ERROR, param->out_buf.ptr (), XCAM_RETURN_ERROR_PARAM,
            "image_handler:%s execute buffer failed, output buffer failed in allocation.",
            XCAM_STR (get_name ()));
    }

    ret = start_work (param);
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret,
        "image_handler(%s) execute buffer failed in starting workers", XCAM_STR (get_name ()));

    return ret;
}

XCamReturn
ImageHandler::finish ()
{
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
ImageHandler::terminate ()
{
    if (_allocator.ptr ())
        _allocator->stop ();

    return XCAM_RETURN_NO_ERROR;
}

void
ImageHandler::execute_status_check (const SmartPtr<ImageHandler::Parameters> &params, const XCamReturn error)
{
    if (_callback.ptr ())
        _callback->execute_status (this, params, error);
}

XCamReturn
ImageHandler::reserve_buffers (const VideoBufferInfo &info, uint32_t count)
{
    XCAM_FAIL_RETURN (
        ERROR, _allocator.ptr (), XCAM_RETURN_ERROR_PARAM,
        "ImageHandler(%s) reserve buffers failed, alloctor was not set", XCAM_STR(get_name ()));

    _allocator->set_video_info (info);

    XCAM_FAIL_RETURN (
        ERROR, _allocator->reserve (count), XCAM_RETURN_ERROR_MEM,
        "ImageHandler(%s) reserve buffers(%d) failed", XCAM_STR(get_name ()), count);

    return XCAM_RETURN_NO_ERROR;
}

SmartPtr<VideoBuffer>
ImageHandler::get_free_buf ()
{
    XCAM_FAIL_RETURN (
        ERROR, _allocator.ptr (), NULL,
        "ImageHandler(%s) get free buffer failed since allocator was not initilized", XCAM_STR(get_name ()));

    return _allocator->get_buffer (_allocator);
}

}
