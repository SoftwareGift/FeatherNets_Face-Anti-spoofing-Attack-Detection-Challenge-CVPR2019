/*
 * vk_geomap_handler.cpp - vulkan geometry map handler implementation
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

#include "vk_geomap_handler.h"
#include "vk_video_buf_allocator.h"
#include "vk_device.h"

#define GEOMAP_SHADER_BINDING_COUNT 5

#define XCAM_VK_GEOMAP_ALIGN_X 4
#define XCAM_VK_GEOMAP_ALIGN_Y 2

namespace XCam {

DECLARE_WORK_CALLBACK (CbGeoMapTask, VKGeoMapHandler, geomap_done);

class VKGeoMapPushConst
    : public VKConstRange::VKPushConstArg
{
public:
    VKGeoMapPushConst (const VKGeoMapHandler::PushConstsProp &prop)
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
    VKGeoMapHandler::PushConstsProp _prop;
};

static const VKShaderInfo geomap_shader_info (
    "main",
std::vector<uint32_t> {
#include "shader_geomap.comp.spv"
});

VKGeoMapHandler::PushConstsProp::PushConstsProp ()
    : in_img_width (0)
    , in_img_height (0)
    , out_img_width (0)
    , out_img_height (0)
    , lut_width (0)
    , lut_height (0)
{
    xcam_mem_clear (lut_step);
    xcam_mem_clear (lut_std_step);
}

VKGeoMapHandler::VKGeoMapHandler (const SmartPtr<VKDevice> dev, const char* name)
    : VKHandler (dev, name)
{
}

bool
VKGeoMapHandler::set_lookup_table (const PointFloat2 *data, uint32_t width, uint32_t height)
{
    XCAM_FAIL_RETURN (
        ERROR, data && width && height, false,
        "VKGeoMapHandler(%s) set look up table failed, data ptr:%p, width:%d, height:%d",
        XCAM_STR (get_name ()), data, width, height);
    XCAM_ASSERT (!_lut_buf.ptr ());

    _lut_width = width;
    _lut_height = height;

    uint32_t lut_size = width * height * 2 * sizeof (float);
    SmartPtr<VKBuffer> buf = VKBuffer::create_buffer (
        get_vk_device (), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, lut_size);
    XCAM_ASSERT (buf.ptr ());

    float *ptr = (float *) buf->map (lut_size, 0);
    XCAM_FAIL_RETURN (ERROR, ptr, false, "VKGeoMapHandler(%s) map range failed", XCAM_STR (get_name ()));
    for (uint32_t i = 0; i < height; ++i) {
        float *ret = &ptr[i * width * 2];
        const PointFloat2 *line = &data[i * width];

        for (uint32_t j = 0; j < width; ++j) {
            ret[j * 2] = line[j].x;
            ret[j * 2 + 1] = line[j].y;
        }
    }
    buf->unmap ();
    _lut_buf = buf;

    return true;
}

bool
VKGeoMapHandler::init_factors ()
{
    XCAM_ASSERT (_lut_width && _lut_height);

    float factor_x, factor_y;
    get_factors (factor_x, factor_y);

    if (!XCAM_DOUBLE_EQUAL_AROUND (factor_x, 0.0f) && !XCAM_DOUBLE_EQUAL_AROUND (factor_y, 0.0f))
        return true;

    return auto_calculate_factors (_lut_width, _lut_height);
}

#define UNIT_BYTES (sizeof (uint32_t))

XCamReturn
VKGeoMapHandler::configure_resource (const SmartPtr<ImageHandler::Parameters> &param)
{
    XCAM_ASSERT (param.ptr () && param->in_buf.ptr ());
    XCAM_FAIL_RETURN (
        ERROR, _lut_buf.ptr (), XCAM_RETURN_ERROR_PARAM,
        "VKGeoMapHandler(%s) configure resource failed, look up table is empty", XCAM_STR (get_name ()));

    const VideoBufferInfo &in_info = param->in_buf->get_video_info ();
    XCAM_FAIL_RETURN (
        ERROR, in_info.format == V4L2_PIX_FMT_NV12, XCAM_RETURN_ERROR_PARAM,
        "VKGeoMapHandler(%s) only support NV12 format, but input format is %s",
        XCAM_STR (get_name ()), xcam_fourcc_to_string (in_info.format));

    uint32_t out_width, out_height;
    get_output_size (out_width, out_height);
    VideoBufferInfo out_info;
    out_info.init (
        in_info.format, out_width, out_height,
        XCAM_ALIGN_UP (out_width, XCAM_VK_GEOMAP_ALIGN_X),
        XCAM_ALIGN_UP (out_height, XCAM_VK_GEOMAP_ALIGN_Y));
    set_out_video_info (out_info);

    init_factors ();

    float factor_x, factor_y;
    get_factors (factor_x, factor_y);
    XCAM_FAIL_RETURN (
        ERROR,
        !XCAM_DOUBLE_EQUAL_AROUND (factor_x, 0.0f) &&
        !XCAM_DOUBLE_EQUAL_AROUND (factor_y, 0.0f),
        XCAM_RETURN_ERROR_PARAM,
        "VKGeoMapHandler(%s) invalid standard factors: x:%f, y:%f",
        XCAM_STR (get_name ()), factor_x, factor_y);

    _image_prop.in_img_width = in_info.aligned_width / UNIT_BYTES;
    _image_prop.in_img_height = in_info.aligned_height;
    _image_prop.out_img_width = out_info.aligned_width / UNIT_BYTES;
    _image_prop.out_img_height = out_info.aligned_height;
    _image_prop.lut_width = _lut_width;
    _image_prop.lut_height = _lut_height;
    _image_prop.lut_std_step[0] = 1.0f / factor_x;
    _image_prop.lut_std_step[1] = 1.0f / factor_y;

    WorkSize global_size (
        XCAM_ALIGN_UP (_image_prop.out_img_width, 8) / 8,
        XCAM_ALIGN_UP (_image_prop.out_img_height, 16) / 16);

    _binding_layout.clear ();
    for (int i = 0; i < GEOMAP_SHADER_BINDING_COUNT; ++i) {
        SmartPtr<VKDescriptor::SetLayoutBinding> binding =
            new VKDescriptor::ComputeLayoutBinding (VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, i);
        _binding_layout.push_back (binding);
    }

    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    if (!_worker.ptr ()) {
        _worker = new VKWorker (get_vk_device(), "VKGeoMapTask", new CbGeoMapTask(this));
        XCAM_ASSERT (_worker.ptr ());

        _worker->set_global_size (global_size);

        VKConstRange::VKPushConstArgs push_consts;
        push_consts.push_back (new VKGeoMapPushConst (_image_prop));
        ret = _worker->build (geomap_shader_info, _binding_layout, push_consts);
        XCAM_FAIL_RETURN (
            ERROR, xcam_ret_is_ok (ret), XCAM_RETURN_ERROR_VULKAN,
            "VKGeoMapHandler(%s) build geomap shader failed.", XCAM_STR (get_name ()));
    }

    return ret;
}

XCamReturn
VKGeoMapHandler::start_work (const SmartPtr<ImageHandler::Parameters> &param)
{
    XCAM_ASSERT (_lut_buf.ptr ());
    XCAM_ASSERT (param.ptr () && param->in_buf.ptr () && param->out_buf.ptr ());
    XCAM_ASSERT (_binding_layout.size () == GEOMAP_SHADER_BINDING_COUNT);

    SmartPtr<VKVideoBuffer> in_vk = param->in_buf.dynamic_cast_ptr<VKVideoBuffer> ();
    SmartPtr<VKVideoBuffer> out_vk = param->out_buf.dynamic_cast_ptr<VKVideoBuffer> ();
    XCAM_FAIL_RETURN (
        ERROR, in_vk.ptr () && out_vk.ptr(), XCAM_RETURN_ERROR_VULKAN,
        "VKGeoMapHandler(%s) param.in_buf or param.out_buf is not vk buffer", XCAM_STR (get_name ()));

    VKDescriptor::SetBindInfoArray bindings (_binding_layout.size ());
    bindings[0].layout = _binding_layout[0];
    bindings[0].desc = VKBufDesc (in_vk->get_vk_buf (), NV12PlaneYIdx);
    bindings[1].layout = _binding_layout[1];
    bindings[1].desc = VKBufDesc (in_vk->get_vk_buf (), NV12PlaneUVIdx);
    bindings[2].layout = _binding_layout[2];
    bindings[2].desc = VKBufDesc (out_vk->get_vk_buf (), NV12PlaneYIdx);
    bindings[3].layout = _binding_layout[3];
    bindings[3].desc = VKBufDesc (out_vk->get_vk_buf (), NV12PlaneUVIdx);
    bindings[4].layout = _binding_layout[4];
    bindings[4].desc = VKBufDesc (_lut_buf);

    float factor_x, factor_y;
    get_factors (factor_x, factor_y);
    _image_prop.lut_step[0] = 1.0f / factor_x;
    _image_prop.lut_step[1] = 1.0f / factor_y;
    _image_prop.lut_step[2] = _image_prop.lut_step[0];
    _image_prop.lut_step[3] = _image_prop.lut_step[1];

    SmartPtr<VKWorker::VKArguments> args = new VKWorker::VKArguments;
    XCAM_ASSERT (args.ptr ());
    args->set_bindings (bindings);
    args->add_push_const (new VKGeoMapPushConst (_image_prop));
    return _worker->work (args);
}

void
VKGeoMapHandler::geomap_done (
    const SmartPtr<Worker> &worker, const SmartPtr<Worker::Arguments> &args, const XCamReturn error)
{
    XCAM_UNUSED (args);
    if (!xcam_ret_is_ok (error)) {
        XCAM_LOG_ERROR ("VKGeoMapHandler(%s) geometry map failed.", XCAM_STR (get_name ()));
    }

    SmartPtr<VKWorker> vk_worker = worker.dynamic_cast_ptr<VKWorker> ();
    XCAM_ASSERT (vk_worker.ptr ());
    vk_worker->wait_fence ();
}

XCamReturn
VKGeoMapHandler::remap (const SmartPtr<VideoBuffer> &in_buf, SmartPtr<VideoBuffer> &out_buf)
{
    SmartPtr<ImageHandler::Parameters> param = new ImageHandler::Parameters (in_buf, out_buf);
    XCAM_ASSERT (param.ptr ());

    XCamReturn ret = execute_buffer (param, false);
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret,
        "VKGeoMapHandler(%s) remap failed", XCAM_STR (get_name ()));

    if (!out_buf.ptr ()) {
        out_buf = param->out_buf;
    }

    return ret;
}

};
