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

namespace XCam {

namespace {

DECLARE_WORK_CALLBACK (CbCopyTask, VKCopyHandler, copy_done);

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
    , out_img_width (0)
    , copy_width (0)
{
}

VKCopyHandler::VKCopyHandler (const SmartPtr<VKDevice> dev, const char* name)
    : VKHandler (dev, name)
{
}

#define UNIT_BYTES (4*sizeof(uint32_t))

XCamReturn
VKCopyHandler::configure_resource (const SmartPtr<ImageHandler::Parameters> &param)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    XCAM_FAIL_RETURN (
        ERROR, param->in_buf.ptr (), XCAM_RETURN_ERROR_VULKAN,
        "VKCopyHandler(%s) param.in_buf is empty.", XCAM_STR (get_name ()));

    VideoBufferInfo out_info = param->in_buf->get_video_info ();
    XCAM_ASSERT (out_info.format == V4L2_PIX_FMT_NV12);
    set_out_video_info (out_info);

    _image_prop.in_img_width =
        _image_prop.out_img_width = out_info.aligned_width / UNIT_BYTES;
    _image_prop.copy_width = out_info.width / UNIT_BYTES;
    WorkSize global_size (
        XCAM_ALIGN_UP (_image_prop.in_img_width, 8 ) / 8,
        XCAM_ALIGN_UP (out_info.height * 3 / 2, 8 ) / 8);

    _binding_layout.clear ();
    for (int i = 0; i < COPY_SHADER_BINDING_COUNT; ++i) {
        SmartPtr<VKDescriptor::SetLayoutBinding> binding =
            new VKDescriptor::ComputeLayoutBinding (VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, i);
        _binding_layout.push_back (binding);
    }

    if (!_worker.ptr ()) {
        _worker = new VKWorker(get_vk_device(), "VKCopyTask", new CbCopyTask(this));
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

    SmartPtr<VKWorker::VKArguments> args = new VKWorker::VKArguments;
    args->set_bindings (bindings);
    args->add_push_const (new VKCopyPushConst (_image_prop));
    return _worker->work (args);
}

void
VKCopyHandler::copy_done (
    const SmartPtr<Worker> &worker,
    const SmartPtr<Worker::Arguments> &args,
    const XCamReturn error)
{
    XCAM_UNUSED (worker);
    XCAM_UNUSED (args);
    if (!xcam_ret_is_ok (error)) {
        XCAM_LOG_ERROR ("VKCopyHandler(%s) copy failed.", XCAM_STR (get_name ()));
    }
    SmartPtr<VKWorker> vk_worker = worker.dynamic_cast_ptr<VKWorker> ();
    XCAM_ASSERT (vk_worker.ptr ());
    vk_worker->wait_fence ();
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
