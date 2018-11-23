/*
 * vk_copy_handler.cpp - vulkan copy handler
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

#include <vulkan/vulkan_std.h>
#include "vk_copy_handler.h"
#include "vk_video_buf_allocator.h"
#include "vk_shader.h"
#include "vk_memory.h"
#include "vk_worker.h"
#include "vk_device.h"

#define COPY_SHADER_BINDING_COUNT 2
#define INVALID_INDEX (uint32_t)(-1)

namespace XCam {

namespace {

DECLARE_WORK_CALLBACK (CbCopyShader, VKCopyHandler, copy_done);

class CopyArgs
    : public VKWorker::VKArguments
{
public:
    explicit CopyArgs (const SmartPtr<ImageHandler::Parameters> &param)
        : _param (param)
    {
        XCAM_ASSERT (param.ptr ());
    }
    const SmartPtr<ImageHandler::Parameters> &get_param () const {
        return _param;
    }

private:
    SmartPtr<ImageHandler::Parameters>    _param;
};

class VKCopyPushConst
    : public VKConstRange::VKPushConstArg
{
public:
    VKCopyPushConst (const VKCopyHandler::PushConstsProp &prop)
        : _prop (prop)
    {}

    bool get_const_data (VkPushConstantRange &range, void *& ptr) {
        range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        range.offset = 0;
        range.size = sizeof (_prop);
        ptr = &_prop;
        return true;
    }

private:
    VKCopyHandler::PushConstsProp _prop;
};

};

static const VKShaderInfo copy_shader_info (
    "main",
std::vector<uint32_t> {
#include "shader_copy.comp.spv"
});

#if 0
VKCopyArguments::VKCopyArguments (const SmartPtr<VKBuffer> in, SmartPtr<VKBuffer> out)
    : _in_buf (in)
    , _out_buf (out)
{
    XCAM_ASSERT (in.ptr());
    XCAM_ASSERT (out.ptr());
}

XCamReturn
VKCopyArguments::prepare_bindings (VKDescriptor::SetBindInfoArray &binding_array)
{
    XCAM_FAIL_RETURN (
        ERROR, _in_buf.ptr () && _out_buf.ptr (), XCAM_RETURN_ERROR_PARAM,
        "VKCopyArguments input or output buffer is empty.");

    binding_array.resize (2);
    binding_array[0].layout = new VKDescriptor::ComputeLayoutBinding (VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 0);
    binding_array[0].desc = VKBufDesc (_in_buf);

    binding_array[0].layout = new VKDescriptor::ComputeLayoutBinding (VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1);
    binding_array[0].desc = VKBufDesc (_out_buf);
    return XCAM_RETURN_NO_ERROR;
}
#endif

VKCopyHandler::PushConstsProp::PushConstsProp ()
    : in_img_width (0)
    , in_x_offset (0)
    , out_img_width (0)
    , out_x_offset (0)
    , copy_width (0)
{
}

VKCopyHandler::VKCopyHandler (const SmartPtr<VKDevice> dev, const char* name)
    : VKHandler (dev, name)
    , _index (INVALID_INDEX)
{
}

bool
VKCopyHandler::set_copy_area (uint32_t idx, const Rect &in_area, const Rect &out_area)
{
    XCAM_FAIL_RETURN (
        ERROR,
        idx != INVALID_INDEX &&
        in_area.width && in_area.height &&
        in_area.width == out_area.width && in_area.height == out_area.height,
        false,
        "VKCopyHandler(%s): set copy area(idx:%d) failed, input size:%dx%d output size:%dx%d",
        XCAM_STR (get_name ()), idx, in_area.width, in_area.height, out_area.width, out_area.height);

    _index = idx;
    _in_area = in_area;
    _out_area = out_area;

    XCAM_LOG_DEBUG ("VKCopyHandler: copy area(idx:%d) input area(%d, %d, %d, %d) output area(%d, %d, %d, %d)",
        idx,
        in_area.pos_x, in_area.pos_y, in_area.width, in_area.height,
        out_area.pos_x, out_area.pos_y, out_area.width, out_area.height);

    return true;
}

#define UNIT_BYTES (4*sizeof(uint32_t))

XCamReturn
VKCopyHandler::configure_resource (const SmartPtr<ImageHandler::Parameters> &param)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    XCAM_ASSERT (param.ptr ());
    XCAM_ASSERT (!_worker.ptr ());

    XCAM_FAIL_RETURN (
        ERROR, param->in_buf.ptr (), XCAM_RETURN_ERROR_VULKAN,
        "VKCopyHandler(%s) param.in_buf is empty.", XCAM_STR (get_name ()));
    XCAM_FAIL_RETURN (
        ERROR, _index != INVALID_INDEX, XCAM_RETURN_ERROR_PARAM,
        "VKCopyHandler(%s) invalid copy area, need set copy area first", XCAM_STR (get_name ()));

    VideoBufferInfo in_info = param->in_buf->get_video_info ();
    VideoBufferInfo out_info;
    if (param->out_buf.ptr ())
        out_info = param->out_buf->get_video_info ();
    else
        out_info = get_out_video_info ();
    XCAM_FAIL_RETURN (
        ERROR, out_info.is_valid (), XCAM_RETURN_ERROR_PARAM,
        "VKCopyHandler(%s) invalid out info.", XCAM_STR (get_name ()));

    _image_prop.in_img_width = in_info.aligned_width / UNIT_BYTES;
    _image_prop.in_x_offset = _in_area.pos_x / UNIT_BYTES;
    _image_prop.out_img_width = out_info.aligned_width / UNIT_BYTES;
    _image_prop.out_x_offset = _out_area.pos_x / UNIT_BYTES;
    _image_prop.copy_width = _in_area.width / UNIT_BYTES;
    WorkSize global_size (
        XCAM_ALIGN_UP (_image_prop.copy_width, 8 ) / 8,
        XCAM_ALIGN_UP (_in_area.height * 3 / 2, 8 ) / 8);

    _binding_layout.clear ();
    for (int i = 0; i < COPY_SHADER_BINDING_COUNT; ++i) {
        SmartPtr<VKDescriptor::SetLayoutBinding> binding =
            new VKDescriptor::ComputeLayoutBinding (VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, i);
        _binding_layout.push_back (binding);
    }

    if (!_worker.ptr ()) {
        _worker = new VKWorker(get_vk_device(), "CbCopyShader", new CbCopyShader (this));
        XCAM_ASSERT (_worker.ptr());

        _worker->set_global_size (global_size);

        VKConstRange::VKPushConstArgs push_consts;
        push_consts.push_back (new VKCopyPushConst (_image_prop));
        ret = _worker->build (copy_shader_info, _binding_layout, push_consts);
        XCAM_FAIL_RETURN (
            ERROR, xcam_ret_is_ok (ret), XCAM_RETURN_ERROR_VULKAN,
            "VKCopyHandler(%s) build copy shader failed.", XCAM_STR (get_name ()));
    }

    return ret;
}

XCamReturn
VKCopyHandler::start_work (const SmartPtr<ImageHandler::Parameters> &param)
{
    XCAM_ASSERT (_binding_layout.size () == COPY_SHADER_BINDING_COUNT);
    SmartPtr<VKVideoBuffer> in_vk = param->in_buf.dynamic_cast_ptr<VKVideoBuffer> ();
    SmartPtr<VKVideoBuffer> out_vk = param->out_buf.dynamic_cast_ptr<VKVideoBuffer> ();

    XCAM_FAIL_RETURN (
        ERROR, in_vk.ptr () && out_vk.ptr(), XCAM_RETURN_ERROR_VULKAN,
        "VKCopyHandler(%s) param.in_buf or param.out_buf is not vk buf.", XCAM_STR (get_name ()));

    VKDescriptor::SetBindInfoArray bindings (_binding_layout.size ());
    bindings[0].layout = _binding_layout[0];
    bindings[0].desc = VKBufDesc (in_vk->get_vk_buf ());
    bindings[1].layout = _binding_layout[1];
    bindings[1].desc = VKBufDesc (out_vk->get_vk_buf ());

    SmartPtr<CopyArgs> args = new CopyArgs (param);
    args->set_bindings (bindings);
    args->add_push_const (new VKCopyPushConst (_image_prop));
    return _worker->work (args);
}

void
VKCopyHandler::copy_done (
    const SmartPtr<Worker> &worker,
    const SmartPtr<Worker::Arguments> &base,
    const XCamReturn error)
{
    if (!xcam_ret_is_ok (error)) {
        XCAM_LOG_ERROR ("VKCopyHandler(%s) copy failed.", XCAM_STR (get_name ()));
    }

    SmartPtr<VKWorker> vk_worker = worker.dynamic_cast_ptr<VKWorker> ();
    XCAM_ASSERT (vk_worker.ptr ());
    vk_worker->wait_fence ();

    SmartPtr<CopyArgs> args = base.dynamic_cast_ptr<CopyArgs> ();
    XCAM_ASSERT (args.ptr ());
    const SmartPtr<ImageHandler::Parameters> param = args->get_param ();
    XCAM_ASSERT (param.ptr ());

    execute_done (param, error);
}

XCamReturn
VKCopyHandler::copy (const SmartPtr<VideoBuffer> &in_buf, SmartPtr<VideoBuffer> &out_buf)
{
    SmartPtr<ImageHandler::Parameters> param = new ImageHandler::Parameters (in_buf, out_buf);
    XCAM_ASSERT (param.ptr ());

    XCamReturn ret = execute_buffer (param, false);
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret,
        "VKCopyHandler(%s) copy failed", XCAM_STR (get_name ()));

    if (!out_buf.ptr ()) {
        out_buf = param->out_buf;
    }

    return ret;
}

};
