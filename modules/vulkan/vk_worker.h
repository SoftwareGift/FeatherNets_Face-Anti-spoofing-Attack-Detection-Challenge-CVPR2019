/*
 * vk_worker.h - vulkan worker class
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

#ifndef XCAM_VK_WORKER_H
#define XCAM_VK_WORKER_H

#include <vulkan/vulkan_std.h>
#include <vulkan/vk_descriptor.h>
#include <worker.h>
#include <string>

namespace XCam {

class VKPipeline;
class VKDevice;
class VKFence;
class VKCmdBuf;

enum VKSahderInfoType {
    VKSahderInfoSpirVBinary = 0,
    VKSahderInfoSpirVPath   = 1,
};

struct VKShaderInfo {
    VKSahderInfoType           type;
    std::string                func_name;
    std::string                spirv_path;
    std::vector<uint32_t>      spirv_bin;

    VKShaderInfo () {}
    VKShaderInfo (const char *func, const char *path)
        : type (VKSahderInfoSpirVPath)
        , func_name (func)
        , spirv_path (path)
    {}
    VKShaderInfo (const char *func, const std::vector<uint32_t> &binary)
        : type (VKSahderInfoSpirVBinary)
        , func_name (func)
        , spirv_bin (binary)
    {}
};

class VKWorker
    : public Worker
{
public:
    class VKArguments:
        public Worker::Arguments
    {
        friend class VKWorker;
    public:
        VKArguments () {}
        VKArguments (VKDescriptor::SetBindInfoArray &arr)
            : _binding_bufs (arr)
        {}
        bool set_bindings (const VKDescriptor::SetBindInfoArray &arr);
        bool add_binding (const VKDescriptor::SetBindInfo &info);
        bool add_push_const (const SmartPtr<VKConstRange::VKPushConstArg> &push_const);

    protected:
        virtual XCamReturn prepare_bindings (
            VKDescriptor::SetBindInfoArray &binding_array,
            VKConstRange::VKPushConstArgs &push_consts);
    private:
        VKDescriptor::SetBindInfoArray       _binding_bufs;
        VKConstRange::VKPushConstArgs        _push_consts;
    };

public:
    explicit VKWorker (SmartPtr<VKDevice> dev, const char *name, const SmartPtr<Callback> &cb = NULL);
    virtual ~VKWorker ();

    XCamReturn build (
        const VKShaderInfo &info,
        const VKDescriptor::BindingArray &bindings,
        const VKConstRange::VKPushConstArgs &consts);

    // derived from Worker
    virtual XCamReturn work (const SmartPtr<Arguments> &args);
    virtual XCamReturn stop ();
    XCamReturn wait_fence ();

private:
    XCAM_DEAD_COPY (VKWorker);

private:
    SmartPtr<VKDevice>             _device;
    SmartPtr<VKDescriptor::Pool>   _desc_pool;
    SmartPtr<VKPipeline>           _pipeline;
    SmartPtr<VKFence>              _fence;
    SmartPtr<VKCmdBuf>             _cmdbuf;
};

}
#endif //XCAM_VK_WORKER_H
