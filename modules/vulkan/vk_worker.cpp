/*
 * vk_worker.cpp - vulkan worker class
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

#include "vk_worker.h"
#include "vk_sync.h"
#include "vk_pipeline.h"
#include "vk_cmdbuf.h"
#include "vk_device.h"
#include "vulkan_common.h"

namespace XCam {

XCamReturn
VKWorker::VKArguments::prepare_bindings (
    VKDescriptor::SetBindInfoArray &binding_array,
    VKConstRange::VKPushConstArgs &push_consts)
{
    XCAM_ASSERT (_binding_bufs.size ());
    XCAM_FAIL_RETURN (
        ERROR, _binding_bufs.size (), XCAM_RETURN_ERROR_PARAM,
        "VKArguments found bindings empty, please check settings or derive interface prepare_bindings");

    binding_array = _binding_bufs;
    push_consts = _push_consts;
    return XCAM_RETURN_NO_ERROR;
}

bool
VKWorker::VKArguments::set_bindings (const VKDescriptor::SetBindInfoArray &arr)
{
    _binding_bufs = arr;
    return true;
}

bool
VKWorker::VKArguments::add_binding (const VKDescriptor::SetBindInfo &info)
{
    _binding_bufs.push_back (info);
    return true;
}

bool
VKWorker::VKArguments::add_push_const (const SmartPtr<VKConstRange::VKPushConstArg> &push_const)
{
    _push_consts.push_back (push_const);
    return true;
}

VKWorker::VKWorker (SmartPtr<VKDevice> dev, const char *name, const SmartPtr<Callback> &cb)
    : Worker (name, cb)
    , _device (dev)
{
}

VKWorker::~VKWorker ()
{
}

XCamReturn
VKWorker::build (
    const VKShaderInfo &info,
    const VKDescriptor::BindingArray &bindings,
    const VKConstRange::VKPushConstArgs &consts)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    XCAM_FAIL_RETURN (
        ERROR, _device.ptr (), XCAM_RETURN_ERROR_VULKAN,
        "vk woker(%s) build failed since vk_device is null.", XCAM_STR (get_name ()));

    SmartPtr<VKShader> shader;
    if (info.type == VKSahderInfoSpirVPath) {
        const char *dir_env = std::getenv (XCAM_VK_SHADER_PATH);
        std::string vk_dir (dir_env, (dir_env ? strlen (dir_env) : 0));
        if (vk_dir.empty () || !vk_dir.length())
            vk_dir = XCAM_DEFAULT_VK_SHADER_PATH;
        std::string spirv_path = vk_dir + "/" + info.spirv_path;
        shader = _device->create_shader (spirv_path.c_str ());
    } else if (info.type == VKSahderInfoSpirVBinary) {
        shader = _device->create_shader (info.spirv_bin);
    }
    XCAM_FAIL_RETURN (
        ERROR, shader.ptr (), XCAM_RETURN_ERROR_VULKAN,
        "vk woker(%s) build failed when creating shader.", XCAM_STR (get_name ()));
    shader->set_func_name (info.func_name.c_str ());

    _desc_pool = new VKDescriptor::Pool (_device);
    XCAM_ASSERT (_desc_pool.ptr ());
    XCAM_FAIL_RETURN (
        ERROR, _desc_pool->add_set_bindings (bindings), XCAM_RETURN_ERROR_VULKAN,
        "vk woker(%s) build failed to add bindings to desc_pool", XCAM_STR (get_name ()));
    ret = _desc_pool->create ();
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret,
        "vk woker(%s) build failed to craete bindings in desc_pool", XCAM_STR (get_name ()));

    VKConstRange::VKConstantArray const_array;
    for (size_t i = 0; i < consts.size(); ++i) {
        VkPushConstantRange data_const = {0, 0, 0};
        void *ptr = NULL;
        consts[i]->get_const_data (data_const, ptr);
        const_array.push_back (data_const);
    }
    _pipeline = VKPipeline::create_compute_pipeline (_device, shader, bindings, const_array);
    XCAM_FAIL_RETURN (
        ERROR, _pipeline.ptr (), XCAM_RETURN_ERROR_VULKAN,
        "vk woker(%s) build failed when creating pipelines.", XCAM_STR (get_name ()));

    _pipeline->set_desc_pool (_desc_pool);

    _cmdbuf = VKCmdBuf::create_command_buffer (_device);
    XCAM_FAIL_RETURN (
        ERROR, _cmdbuf.ptr (), XCAM_RETURN_ERROR_VULKAN,
        "vk woker(%s) build failed when creating command buffers.", XCAM_STR (get_name ()));

    _fence = _device->create_fence (VK_FENCE_CREATE_SIGNALED_BIT);
    XCAM_FAIL_RETURN (
        ERROR, _fence.ptr (), XCAM_RETURN_ERROR_VULKAN,
        "vk woker(%s) build failed when creating fence.", XCAM_STR (get_name ()));
    return XCAM_RETURN_NO_ERROR;
}

// derived from Worker
XCamReturn
VKWorker::work (const SmartPtr<Worker::Arguments> &args)
{
    SmartPtr<VKArguments> vk_args = args.dynamic_cast_ptr<VKArguments>();
    XCAM_FAIL_RETURN (
        ERROR, vk_args.ptr(), XCAM_RETURN_ERROR_PARAM,
        "vk woker(%s) work argements error.", XCAM_STR (get_name ()));

    VKDescriptor::SetBindInfoArray binding_array;
    VKConstRange::VKPushConstArgs push_consts;
    XCamReturn ret = vk_args->prepare_bindings (binding_array, push_consts);

    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret,
        "vk woker(%s) prepare argements failed.", XCAM_STR (get_name ()));

    XCAM_FAIL_RETURN (
        ERROR, !binding_array.empty (), XCAM_RETURN_ERROR_PARAM,
        "vk woker(%s) binding_array is empty.", XCAM_STR (get_name ()));

    ret = _pipeline->update_bindings (binding_array);

    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret,
        "vk woker(%s) update binding argements failed.", XCAM_STR (get_name ()));

    const WorkSize global = get_global_size ();
    SmartPtr<VKCmdBuf::DispatchParam> dispatch  =
        new VKCmdBuf::DispatchParam (_pipeline, global.value[0], global.value[1], global.value[2]);
    if (!push_consts.empty()) {
        XCAM_FAIL_RETURN (
            ERROR, dispatch->update_push_consts (push_consts), XCAM_RETURN_ERROR_PARAM,
            "vk woker(%s) update push-consts failed.", XCAM_STR (get_name ()));
    }

    ret = _cmdbuf->record (dispatch);
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret,
        "vk woker(%s) record cmdbuf failed.", XCAM_STR (get_name ()));

    ret = _device->compute_queue_submit (_cmdbuf, _fence);
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret,
        "vk woker(%s) submit compute queue failed.", XCAM_STR (get_name ()));

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
VKWorker::stop ()
{
    if (_pipeline.ptr () && _device.ptr ()) {
        if (_fence.ptr ()) {
            _fence->wait ();
            _fence->reset ();
        }
        _device->compute_queue_wait_idle ();
    }
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
VKWorker::wait_fence ()
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    if (_fence.ptr ()) {
        ret = _fence->wait ();
        if (xcam_ret_is_ok (ret)) {
            XCAM_LOG_ERROR ("vk woker(%s) wait fence failed.", XCAM_STR (get_name ()));
        }
        _fence->reset ();
    }

    return ret;
}

}
