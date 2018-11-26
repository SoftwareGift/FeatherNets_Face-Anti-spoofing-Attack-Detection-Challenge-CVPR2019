/*
 * vk_blender.cpp - vulkan blender implementation
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

#include "xcam_utils.h"

#include "vk_device.h"
#include "vk_worker.h"
#include "vk_blender.h"
#include "vk_video_buf_allocator.h"

#define DUMP_BUFFER 0

#define GAUSS_RADIUS 2
#define GAUSS_DIAMETER ((GAUSS_RADIUS)*2+1)

const float gauss_coeffs[GAUSS_DIAMETER] = {0.152f, 0.222f, 0.252f, 0.222f, 0.152f};

#define GS_SHADER_BINDING_COUNT          4
#define LAP_SHADER_BINDING_COUNT         6
#define BLEND_SHADER_BINDING_COUNT       7
#define RECONSTRUCT_SHADER_BINDING_COUNT 9

#define CHECK_RET(ret, format, ...) \
    if (!xcam_ret_is_ok (ret)) { \
        XCAM_LOG_ERROR (format, ## __VA_ARGS__); \
    }

#define DECLARE_VK_PUSH_CONST(PushConstClass, PushConstsProp) \
    class PushConstClass : public VKConstRange::VKPushConstArg { \
    private: PushConstsProp    _prop; \
    public: \
        PushConstClass (const PushConstsProp &prop) : _prop (prop) {} \
        bool get_const_data (VkPushConstantRange &range, void *& ptr) { \
            range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; \
            range.offset = 0; \
            range.size = sizeof (_prop); \
            ptr = &_prop; \
            return true; } \
    }

namespace XCam {

#if DUMP_BUFFER
static void
dump_vkbuf_with_perfix (const SmartPtr<VKBuffer> &buf, const char *perfix_name)
{
    XCAM_ASSERT (buf.ptr ());
    XCAM_ASSERT (perfix_name);

    const VKBufInfo &info = buf->get_buf_info ();
    char file_name[XCAM_VK_NAME_LENGTH];
    snprintf (
        file_name, XCAM_VK_NAME_LENGTH, "%s-%dx%d.%s",
        perfix_name, info.width, info.height, xcam_fourcc_to_string (info.format));

    FILE *fp = fopen (file_name, "wb");
    if (!fp) {
        XCAM_LOG_ERROR ( "vk-blend open file(%s) failed", file_name);
    }

    uint8_t *ptr = (uint8_t *)buf->map ();
    for (uint32_t i = 0; i < info.height * 3 / 2; ++i) {
        uint8_t *start = ptr + info.aligned_width * i;
        fwrite (start, info.width, 1, fp);
    }
    buf->unmap ();
    fclose (fp);
}
#define dump_vkbuf dump_vkbuf_with_perfix

static void
dump_level_vkbuf (const SmartPtr<VKBuffer> &buf, const char *name, uint32_t level, uint32_t idx)
{
    char file_name[XCAM_VK_NAME_LENGTH];
    snprintf (file_name, XCAM_VK_NAME_LENGTH, "%s-L%d-Idx%d", name, level, idx);

    dump_vkbuf_with_perfix (buf, file_name);
}
#endif

namespace VKBlenderPriv {

enum ShaderID {
    ShaderGaussScalePyr = 0,
    ShaderLapTransPyr,
    ShaderBlendPyr,
    ShaderReconstructPyr
};

static const VKShaderInfo shaders_info[] = {
    VKShaderInfo (
    "main",
    std::vector<uint32_t> {
#include "shader_gauss_scale_pyr.comp.spv"
    }),
    VKShaderInfo (
    "main",
    std::vector<uint32_t> {
#include "shader_lap_trans_pyr.comp.spv"
    }),
    VKShaderInfo (
    "main",
    std::vector<uint32_t> {
#include "shader_blend_pyr.comp.spv"
    }),
    VKShaderInfo (
    "main",
    std::vector<uint32_t> {
#include "shader_reconstruct_pyr.comp.spv"
    })
};

struct GaussScalePushConstsProp {
    uint     in_img_width;
    uint     in_img_height;
    uint     in_offset_x;
    uint     out_img_width;
    uint     out_img_height;
    uint     merge_width;

    GaussScalePushConstsProp ()
        : in_img_width (0)
        , in_img_height (0)
        , in_offset_x (0)
        , out_img_width (0)
        , out_img_height (0)
        , merge_width (0)
    {}
};

struct LapPushConstsProp {
    uint     in_img_width;
    uint     in_img_height;
    uint     in_offset_x;
    uint     gaussscale_img_width;
    uint     gaussscale_img_height;
    uint     merge_width;

    LapPushConstsProp ()
        : in_img_width (0)
        , in_img_height (0)
        , in_offset_x (0)
        , gaussscale_img_width (0)
        , gaussscale_img_height (0)
        , merge_width (0)
    {}
};

struct BlendPushConstsProp {
    uint     in_img_width;

    BlendPushConstsProp ()
        : in_img_width (0)
    {}
};

struct ReconstructPushConstsProp {
    uint     lap_img_width;
    uint     lap_img_height;
    uint     out_img_width;
    uint     out_offset_x;
    uint     prev_blend_img_width;
    uint     prev_blend_img_height;

    ReconstructPushConstsProp ()
        : lap_img_width (0)
        , lap_img_height (0)
        , out_img_width (0)
        , out_offset_x (0)
        , prev_blend_img_width (0)
        , prev_blend_img_height (0)
    {}
};

DECLARE_VK_PUSH_CONST (VKGaussScalePushConst, GaussScalePushConstsProp);
DECLARE_VK_PUSH_CONST (VKLapPushConst, LapPushConstsProp);
DECLARE_VK_PUSH_CONST (VKBlendPushConst, BlendPushConstsProp);
DECLARE_VK_PUSH_CONST (VKReconstructPushConst, ReconstructPushConstsProp);

DECLARE_WORK_CALLBACK (CbGaussScalePyr, VKBlender, gauss_scale_done);
DECLARE_WORK_CALLBACK (CbLapTransPyr, VKBlender, lap_trans_done);
DECLARE_WORK_CALLBACK (CbBlendPyr, VKBlender, blend_done);
DECLARE_WORK_CALLBACK (CbReconstructPyr, VKBlender, reconstruct_done);

class BlendArgs
    : public VKWorker::VKArguments
{
public:
    BlendArgs (uint32_t lv, VKBlender::BufIdx i = VKBlender::BufIdx0);

    uint32_t get_level () {
        return _level;
    }
    VKBlender::BufIdx get_idx () {
        return _idx;
    }

private:
    uint32_t             _level;
    VKBlender::BufIdx    _idx;
};

struct PyrLayer {
    uint32_t                                blend_width;
    uint32_t                                blend_height;

    SmartPtr<VKBlender::Sync>               lap_sync[VKBlender::BufIdxMax];
    SmartPtr<VKBlender::Sync>               blend_sync;
    SmartPtr<VKBlender::Sync>               reconstruct_sync;

    SmartPtr<VKBuffer>                      gs_buf[VKBlender::BufIdxMax];
    SmartPtr<VKBuffer>                      lap_buf[VKBlender::BufIdxMax];
    SmartPtr<VKBuffer>                      mask;
    SmartPtr<VKBuffer>                      blend_buf;
    SmartPtr<VKBuffer>                      reconstruct_buf;

    VKDescriptor::SetBindInfoArray          gs_bindings[VKBlender::BufIdxMax];
    VKDescriptor::SetBindInfoArray          lap_bindings[VKBlender::BufIdxMax];
    VKDescriptor::SetBindInfoArray          blend_bindings;
    VKDescriptor::SetBindInfoArray          reconstruct_bindings;

    SmartPtr<VKConstRange::VKPushConstArg>  gs_consts[VKBlender::BufIdxMax];
    SmartPtr<VKConstRange::VKPushConstArg>  lap_consts[VKBlender::BufIdxMax];
    SmartPtr<VKConstRange::VKPushConstArg>  blend_consts;
    SmartPtr<VKConstRange::VKPushConstArg>  reconstruct_consts;

    WorkSize                                gs_global_size[VKBlender::BufIdxMax];
    WorkSize                                lap_global_size[VKBlender::BufIdxMax];
    WorkSize                                blend_global_size;
    WorkSize                                reconstruct_global_size;

    VKWorker                               *gauss_scale[VKBlender::BufIdxMax];
    VKWorker                               *lap_trans[VKBlender::BufIdxMax];
    VKWorker                               *blend;
    VKWorker                               *reconstruct;

    PyrLayer ();
};

typedef std::map<ShaderID, SmartPtr<VKWorker>>    VKWorkers;

class BlenderImpl {
public:
    PyrLayer                pyr_layer[XCAM_VK_MAX_LEVEL];
    uint32_t                pyr_layers_num;

private:
    VKBlender              *_blender;
    VKWorkers               _workers;

public:
    BlenderImpl (VKBlender *blender, uint32_t layers_num)
        : pyr_layers_num (layers_num)
        , _blender (blender)
    {
        XCAM_ASSERT (layers_num >= 2 && layers_num <= XCAM_VK_MAX_LEVEL);
    }

    XCamReturn start_gauss_scale (uint32_t level, VKBlender::BufIdx idx);
    XCamReturn start_lap_trans (uint32_t level, VKBlender::BufIdx idx);
    XCamReturn start_top_blend ();
    XCamReturn start_reconstruct (uint32_t level);
    XCamReturn stop ();

    void init_syncs ();
    XCamReturn init_layers_bufs (const SmartPtr<ImageHandler::Parameters> &base);
    XCamReturn bind_io_bufs_to_layer0 (
        SmartPtr<VideoBuffer> &input0, SmartPtr<VideoBuffer> &input1, SmartPtr<VideoBuffer> &output);
    XCamReturn bind_io_vkbufs_to_desc ();
    XCamReturn fix_parameters ();
    XCamReturn create_workers (const SmartPtr<VKBlender> &blender);
    XCamReturn redirect_workers ();

private:
    XCamReturn start_lap_tran (uint32_t level, VKBlender::BufIdx idx);

    XCamReturn layer0_allocate_bufs (SmartPtr<VKDevice> dev);
    XCamReturn layer0_init_mask (SmartPtr<VKDevice> dev);

    XCamReturn layerx_allocate_bufs (SmartPtr<VKDevice> dev, uint32_t level);
    XCamReturn allocate_vk_bufs (SmartPtr<VKDevice> dev, uint32_t level);
    XCamReturn scale_down_mask (SmartPtr<VKDevice> dev, uint32_t level);

    XCamReturn fix_gs_params (uint32_t level, VKBlender::BufIdx idx);
    XCamReturn fix_lap_trans_params (uint32_t level, VKBlender::BufIdx idx);
    XCamReturn fix_blend_params ();
    XCamReturn fix_reconstruct_params (uint32_t level);
};

BlendArgs::BlendArgs (uint32_t lv, VKBlender::BufIdx i)
    : _level (lv)
    , _idx (i)
{
    XCAM_ASSERT (lv < XCAM_VK_DEFAULT_LEVEL);
    XCAM_ASSERT (i < VKBlender::BufIdxMax);
}

PyrLayer::PyrLayer ()
    : blend_width (0)
    , blend_height (0)
{
}

void
BlenderImpl::init_syncs ()
{
    for (uint32_t i = 0; i < pyr_layers_num - 1; ++i) {
        PyrLayer &layer = pyr_layer[i];

        layer.lap_sync[VKBlender::BufIdx0] = new VKBlender::Sync (2);
        XCAM_ASSERT (layer.lap_sync[VKBlender::BufIdx0].ptr ());
        layer.lap_sync[VKBlender::BufIdx1] = new VKBlender::Sync (2);
        XCAM_ASSERT (layer.lap_sync[VKBlender::BufIdx1].ptr ());

        layer.reconstruct_sync = new VKBlender::Sync (3);
        XCAM_ASSERT (layer.reconstruct_sync.ptr ());
    }

    pyr_layer[pyr_layers_num - 1].blend_sync = new VKBlender::Sync (2);
    XCAM_ASSERT (pyr_layer[pyr_layers_num - 1].blend_sync.ptr ());
}

XCamReturn
BlenderImpl::bind_io_bufs_to_layer0 (
    SmartPtr<VideoBuffer> &input0, SmartPtr<VideoBuffer> &input1, SmartPtr<VideoBuffer> &output)
{
    XCAM_ASSERT (input0.ptr () && input1.ptr ());

    SmartPtr<VKVideoBuffer> in0_vk = input0.dynamic_cast_ptr<VKVideoBuffer> ();
    SmartPtr<VKVideoBuffer> in1_vk = input1.dynamic_cast_ptr<VKVideoBuffer> ();

    PyrLayer &layer0 = pyr_layer[0];
    layer0.gs_buf[VKBlender::BufIdx0] = in0_vk->get_vk_buf ();
    layer0.gs_buf[VKBlender::BufIdx1] = in1_vk->get_vk_buf ();
    XCAM_ASSERT (layer0.gs_buf[VKBlender::BufIdx0].ptr () && layer0.gs_buf[VKBlender::BufIdx1].ptr ());

    if (!output.ptr ())
        return XCAM_RETURN_NO_ERROR;

    SmartPtr<VKVideoBuffer> out_vk = output.dynamic_cast_ptr<VKVideoBuffer> ();
    XCAM_ASSERT (out_vk.ptr ());

    layer0.reconstruct_buf = out_vk->get_vk_buf ();
    XCAM_ASSERT (layer0.reconstruct_buf.ptr ());
    layer0.blend_buf = layer0.reconstruct_buf;

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
BlenderImpl::bind_io_vkbufs_to_desc ()
{
    PyrLayer &layer0 = pyr_layer[0];
    PyrLayer &layer1 = pyr_layer[1];
    XCAM_ASSERT (layer0.gs_buf[VKBlender::BufIdx0].ptr () && layer0.gs_buf[VKBlender::BufIdx1].ptr ());
    XCAM_ASSERT (layer0.reconstruct_buf.ptr ());

    VKDescriptor::SetBindInfoArray &gs_bindings0 = layer1.gs_bindings[VKBlender::BufIdx0];
    VKDescriptor::SetBindInfoArray &gs_bindings1 = layer1.gs_bindings[VKBlender::BufIdx1];
    gs_bindings0[0].desc = VKBufDesc (layer0.gs_buf[VKBlender::BufIdx0], NV12PlaneYIdx);
    gs_bindings0[1].desc = VKBufDesc (layer0.gs_buf[VKBlender::BufIdx0], NV12PlaneUVIdx);
    gs_bindings1[0].desc = VKBufDesc (layer0.gs_buf[VKBlender::BufIdx1], NV12PlaneYIdx);
    gs_bindings1[1].desc = VKBufDesc (layer0.gs_buf[VKBlender::BufIdx1], NV12PlaneUVIdx);

    VKDescriptor::SetBindInfoArray &lap_bindings0 = layer0.lap_bindings[VKBlender::BufIdx0];
    VKDescriptor::SetBindInfoArray &lap_bindings1 = layer0.lap_bindings[VKBlender::BufIdx1];
    lap_bindings0[0].desc = VKBufDesc (layer0.gs_buf[VKBlender::BufIdx0], NV12PlaneYIdx);
    lap_bindings0[1].desc = VKBufDesc (layer0.gs_buf[VKBlender::BufIdx0], NV12PlaneUVIdx);
    lap_bindings1[0].desc = VKBufDesc (layer0.gs_buf[VKBlender::BufIdx1], NV12PlaneYIdx);
    lap_bindings1[1].desc = VKBufDesc (layer0.gs_buf[VKBlender::BufIdx1], NV12PlaneUVIdx);

    layer0.reconstruct_bindings[4].desc = VKBufDesc (layer0.reconstruct_buf, NV12PlaneYIdx);
    layer0.reconstruct_bindings[5].desc = VKBufDesc (layer0.reconstruct_buf, NV12PlaneUVIdx);

    return XCAM_RETURN_NO_ERROR;
}

static void
convert_to_vkinfo (const VideoBufferInfo &info, VKBufInfo &vk_info)
{
    vk_info.format = info.format;
    vk_info.width = info.width;
    vk_info.height = info.height;
    vk_info.aligned_width = info.aligned_width;
    vk_info.aligned_height = info.aligned_height;
    vk_info.size = info.size;
    vk_info.strides[0] = info.strides[0];
    vk_info.strides[1] = info.strides[1];
    vk_info.offsets[0] = info.offsets[0];
    vk_info.offsets[1] = info.offsets[1];
    vk_info.slice_size[0] = info.strides[0] * info.aligned_height;
    vk_info.slice_size[1] = info.size - info.offsets[1];
}

XCamReturn
BlenderImpl::layer0_allocate_bufs (SmartPtr<VKDevice> dev)
{
    if (pyr_layers_num == 1)
        return XCAM_RETURN_NO_ERROR;

    PyrLayer &layer0 = pyr_layer[0];
    XCAM_FAIL_RETURN (
        ERROR, layer0.blend_width && layer0.blend_height, XCAM_RETURN_ERROR_PARAM,
        "vk-blend invalid blend size:%dx%d", layer0.blend_width, layer0.blend_height);

    VideoBufferInfo info;
    info.init (
        V4L2_PIX_FMT_NV12, layer0.blend_width, layer0.blend_height,
        XCAM_ALIGN_UP (layer0.blend_width, VK_BLENDER_ALIGN_X),
        XCAM_ALIGN_UP (layer0.blend_height, VK_BLENDER_ALIGN_X));

    VKBufInfo vk_info;
    convert_to_vkinfo (info, vk_info);

    for (int idx = 0; idx < VKBlender::BufIdxMax; ++idx) {
        layer0.lap_buf[idx] = VKBuffer::create_buffer (dev, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, vk_info.size);
        XCAM_ASSERT (layer0.lap_buf[idx].ptr ());
        layer0.lap_buf[idx]->set_buf_info (vk_info);
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
BlenderImpl::allocate_vk_bufs (SmartPtr<VKDevice> dev, uint32_t level)
{
    XCAM_ASSERT (level >= 1 && level < pyr_layers_num);

    PyrLayer &layer = pyr_layer[level];
    VideoBufferInfo info;
    info.init (
        V4L2_PIX_FMT_NV12, layer.blend_width, layer.blend_height,
        XCAM_ALIGN_UP (layer.blend_width, VK_BLENDER_ALIGN_X),
        XCAM_ALIGN_UP (layer.blend_height, VK_BLENDER_ALIGN_X));

    VKBufInfo vk_info;
    convert_to_vkinfo (info, vk_info);

    bool top_layer = (level == pyr_layers_num - 1);
    for (int idx = 0; idx < VKBlender::BufIdxMax; ++idx) {
        layer.gs_buf[idx] = VKBuffer::create_buffer (dev, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, vk_info.size);
        XCAM_ASSERT (layer.gs_buf[idx].ptr ());
        layer.gs_buf[idx]->set_buf_info (vk_info);

        if (top_layer)
            continue;

        layer.lap_buf[idx] = VKBuffer::create_buffer (dev, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, vk_info.size);
        XCAM_ASSERT (layer.lap_buf[idx].ptr ());
        layer.lap_buf[idx]->set_buf_info (vk_info);
    }

    layer.reconstruct_buf = VKBuffer::create_buffer (dev, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, vk_info.size);
    XCAM_ASSERT (layer.reconstruct_buf.ptr ());
    layer.reconstruct_buf->set_buf_info (vk_info);

    if (top_layer)
        layer.blend_buf = layer.reconstruct_buf;

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
BlenderImpl::layer0_init_mask (SmartPtr<VKDevice> dev)
{
    PyrLayer &layer = pyr_layer[0];
    XCAM_ASSERT (layer.blend_width && ((layer.blend_width % VK_BLENDER_ALIGN_X) == 0));

    uint32_t buf_size = layer.blend_width * sizeof (uint8_t);
    SmartPtr<VKBuffer> buf = VKBuffer::create_buffer (dev, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, buf_size);
    XCAM_ASSERT (buf.ptr ());

    VKBufInfo info;
    info.width = layer.blend_width;
    info.height = 1;
    info.size = buf_size;
    buf->set_buf_info (info);

    std::vector<float> gauss_table;
    uint32_t quater = info.width / 4;

    get_gauss_table (quater, (quater + 1) / 4.0f, gauss_table, false);
    for (uint32_t i = 0; i < gauss_table.size (); ++i) {
        float value = ((i < quater) ? (128.0f * (2.0f - gauss_table[i])) : (128.0f * gauss_table[i]));
        value = XCAM_CLAMP (value, 0.0f, 255.0f);
        gauss_table[i] = value;
    }

    uint8_t *mask_ptr = (uint8_t *) buf->map (buf_size, 0);
    XCAM_FAIL_RETURN (ERROR, mask_ptr, XCAM_RETURN_ERROR_PARAM, "vk-blend map range failed");

    uint32_t gauss_start_pos = (info.width - gauss_table.size ()) / 2;
    uint32_t idx = 0;
    for (idx = 0; idx < gauss_start_pos; ++idx) {
        mask_ptr[idx] = 255;
    }
    for (uint32_t i = 0; i < gauss_table.size (); ++idx, ++i) {
        mask_ptr[idx] = (uint8_t) gauss_table[i];
    }
    for (; idx < info.width; ++idx) {
        mask_ptr[idx] = 0;
    }
    buf->unmap ();

    layer.mask = buf;

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
BlenderImpl::scale_down_mask (SmartPtr<VKDevice> dev, uint32_t level)
{
    XCAM_ASSERT (level >= 1 && level < pyr_layers_num);

    PyrLayer &layer = pyr_layer[level];
    PyrLayer &prev_layer = pyr_layer[level - 1];

    XCAM_ASSERT (prev_layer.mask.ptr ());
    XCAM_ASSERT (layer.blend_width && ((layer.blend_width % VK_BLENDER_ALIGN_X) == 0));

    uint32_t buf_size = layer.blend_width * sizeof (uint8_t);
    SmartPtr<VKBuffer> buf =  VKBuffer::create_buffer (dev, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, buf_size);
    XCAM_ASSERT (buf.ptr ());

    VKBufInfo info;
    info.width = layer.blend_width;
    info.height = 1;
    info.size = buf_size;
    buf->set_buf_info (info);

    const VKBufInfo prev_info = prev_layer.mask->get_buf_info ();
    uint8_t *prev_ptr = (uint8_t *) prev_layer.mask->map (prev_info.size, 0);
    XCAM_FAIL_RETURN (ERROR, prev_ptr, XCAM_RETURN_ERROR_PARAM, "vk-blend map range failed");

    uint8_t *cur_ptr = (uint8_t *) buf->map (info.size, 0);
    XCAM_FAIL_RETURN (ERROR, cur_ptr, XCAM_RETURN_ERROR_PARAM, "vk-blend map range failed");

    for (uint32_t i = 0; i < info.width; ++i) {
        int prev_start = i * 2 - 2;
        float sum = 0.0f;

        for (int j = 0; j < GAUSS_DIAMETER; ++j) {
            int prev_idx = XCAM_CLAMP (prev_start + j, 0, (int)prev_info.width);
            sum += prev_ptr[prev_idx] * gauss_coeffs[j];
        }

        cur_ptr[i] = XCAM_CLAMP (sum, 0.0f, 255.0f);
    }

    buf->unmap ();
    prev_layer.mask->unmap ();

    layer.mask = buf;

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
BlenderImpl::layerx_allocate_bufs (SmartPtr<VKDevice> dev, uint32_t level)
{
    XCAM_ASSERT (level >= 1 && level < pyr_layers_num);

    PyrLayer &layer = pyr_layer[level];
    PyrLayer &prev_layer = pyr_layer[level - 1];
    XCAM_FAIL_RETURN (
        ERROR, prev_layer.blend_width && prev_layer.blend_height, XCAM_RETURN_ERROR_PARAM,
        "vk-blend invalid blend size:%dx%d", prev_layer.blend_width, prev_layer.blend_height);

    layer.blend_width =  XCAM_ALIGN_UP ((prev_layer.blend_width + 1) / 2, VK_BLENDER_ALIGN_X);
    layer.blend_height = XCAM_ALIGN_UP ((prev_layer.blend_height + 1) / 2, VK_BLENDER_ALIGN_Y);

    XCamReturn ret = allocate_vk_bufs (dev, level);
    XCAM_FAIL_RETURN (ERROR, xcam_ret_is_ok (ret), ret, "vk-blend build vk buffers failed, level:%d", level);

    ret = scale_down_mask (dev, level);
    XCAM_FAIL_RETURN (ERROR, xcam_ret_is_ok (ret), ret, "vk-blend scale down mask failed, level:%d", level);

    return ret;
}

static XCamReturn
check_desc (
    const VideoBufferInfo &in0_info, const VideoBufferInfo &in1_info,
    const Rect &merge0_area, const Rect &merge1_area)
{
    XCAM_FAIL_RETURN (
        ERROR,
        in0_info.width && in0_info.height && in1_info.width &&
        in0_info.height == in1_info.height,
        XCAM_RETURN_ERROR_PARAM,
        "vk-blend invalid buffer size: in0:%dx%d in1:%dx%d out:%dx%d",
        in0_info.width, in0_info.height, in1_info.width, in1_info.height);

    XCAM_FAIL_RETURN (
        ERROR,
        merge0_area.width && merge0_area.width == merge1_area.width &&
        merge0_area.pos_y == 0 && merge1_area.pos_y == 0 &&
        merge0_area.height == merge1_area.height && merge0_area.height == (int32_t)in0_info.height,
        XCAM_RETURN_ERROR_PARAM,
        "vk-blend invalid merge area: merge0(%d, %d, %d, %d) merge1(%d, %d, %d, %d)",
        merge0_area.pos_x, merge0_area.pos_y, merge0_area.width, merge0_area.height,
        merge1_area.pos_x, merge1_area.pos_y, merge1_area.width, merge1_area.height);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
BlenderImpl::init_layers_bufs (const SmartPtr<ImageHandler::Parameters> &base)
{
    XCAM_ASSERT (base.ptr ());
    SmartPtr<VKBlender::BlenderParam> param = base.dynamic_cast_ptr<VKBlender::BlenderParam> ();
    XCAM_ASSERT (param.ptr () && param->in_buf.ptr () && param->in1_buf.ptr ());

    const VideoBufferInfo &in0_info = param->in_buf->get_video_info ();
    const VideoBufferInfo &in1_info = param->in1_buf->get_video_info ();
    const Rect merge0_area = _blender->get_input_merge_area (VKBlender::BufIdx0);
    const Rect merge1_area = _blender->get_input_merge_area (VKBlender::BufIdx1);

    XCamReturn ret = check_desc (in0_info, in1_info, merge0_area, merge1_area);
    XCAM_FAIL_RETURN (ERROR, xcam_ret_is_ok (ret), ret, "vk-blend check desc failed");

    PyrLayer &layer0 = pyr_layer[0];
    layer0.blend_width = XCAM_ALIGN_UP (merge0_area.width, VK_BLENDER_ALIGN_X);
    layer0.blend_height = XCAM_ALIGN_UP (merge0_area.height, VK_BLENDER_ALIGN_Y);

    SmartPtr<VKDevice> dev = _blender->get_vk_device ();
    XCAM_ASSERT (dev.ptr ());

    ret = bind_io_bufs_to_layer0 (param->in_buf, param->in1_buf, param->out_buf);
    XCAM_FAIL_RETURN (ERROR, xcam_ret_is_ok (ret), ret, "vk-blend bind bufs to layer0 failed");
    ret = layer0_allocate_bufs (dev);
    XCAM_FAIL_RETURN (ERROR, xcam_ret_is_ok (ret), ret, "vk-blend layer0 build buffers failed");
    ret = layer0_init_mask (dev);
    XCAM_FAIL_RETURN (ERROR, xcam_ret_is_ok (ret), ret, "vk-blend layer0 init mask failed");

    for (uint32_t level = 1; level < pyr_layers_num; ++level) {
        layerx_allocate_bufs (dev, level);
        XCAM_FAIL_RETURN (
            ERROR, xcam_ret_is_ok (ret), ret,
            "vk-blend build buffers failed, level:%d", level);
    }

    return XCAM_RETURN_NO_ERROR;
}

static SmartPtr<VKWorker>
create_gauss_scale_pyr_shader (const SmartPtr<VKBlender> &blender)
{
    SmartPtr<VKDevice> dev = blender->get_vk_device ();
    XCAM_ASSERT (dev.ptr ());

    GaussScalePushConstsProp prop;
    VKConstRange::VKPushConstArgs push_consts;
    push_consts.push_back (new VKGaussScalePushConst (prop));

    VKDescriptor::BindingArray binding_layout;
    binding_layout.clear ();
    for (int i = 0; i < GS_SHADER_BINDING_COUNT; ++i) {
        SmartPtr<VKDescriptor::SetLayoutBinding> binding =
            new VKDescriptor::ComputeLayoutBinding (VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, i);
        binding_layout.push_back (binding);
    }

    SmartPtr<VKWorker> worker = new VKWorker (dev, "VKGaussScaleShader", new CbGaussScalePyr (blender));
    XCAM_ASSERT (worker.ptr ());

    XCamReturn ret = worker->build (shaders_info[ShaderGaussScalePyr], binding_layout, push_consts);
    XCAM_FAIL_RETURN (ERROR, xcam_ret_is_ok (ret), NULL, "vk-blend build VKGaussScaleShader failed");

    return worker;
}

static SmartPtr<VKWorker>
create_lap_trans_pyr_shader (const SmartPtr<VKBlender> &blender)
{
    SmartPtr<VKDevice> dev = blender->get_vk_device ();
    XCAM_ASSERT (dev.ptr ());

    LapPushConstsProp prop;
    VKConstRange::VKPushConstArgs push_consts;
    push_consts.push_back (new VKLapPushConst (prop));

    VKDescriptor::BindingArray binding_layout;
    binding_layout.clear ();
    for (int i = 0; i < LAP_SHADER_BINDING_COUNT; ++i) {
        SmartPtr<VKDescriptor::SetLayoutBinding> binding =
            new VKDescriptor::ComputeLayoutBinding (VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, i);
        binding_layout.push_back (binding);
    }

    SmartPtr<VKWorker> worker = new VKWorker (dev, "VKLapTransShader", new CbLapTransPyr (blender));
    XCAM_ASSERT (worker.ptr ());

    XCamReturn ret = worker->build (shaders_info[ShaderLapTransPyr], binding_layout, push_consts);
    XCAM_FAIL_RETURN (ERROR, xcam_ret_is_ok (ret), NULL, "vk-blend build VKLapTransShader failed");

    return worker;
}

static SmartPtr<VKWorker>
create_blend_pyr_shader (const SmartPtr<VKBlender> &blender)
{
    SmartPtr<VKDevice> dev = blender->get_vk_device ();
    XCAM_ASSERT (dev.ptr ());

    BlendPushConstsProp prop;
    VKConstRange::VKPushConstArgs push_consts;
    push_consts.push_back (new VKBlendPushConst (prop));

    VKDescriptor::BindingArray binding_layout;
    binding_layout.clear ();
    for (int i = 0; i < BLEND_SHADER_BINDING_COUNT; ++i) {
        SmartPtr<VKDescriptor::SetLayoutBinding> binding =
            new VKDescriptor::ComputeLayoutBinding (VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, i);
        binding_layout.push_back (binding);
    }

    SmartPtr<VKWorker> worker = new VKWorker (dev, "VKBlendPyrShader", new CbBlendPyr (blender));
    XCAM_ASSERT (worker.ptr ());

    XCamReturn ret = worker->build (shaders_info[ShaderBlendPyr], binding_layout, push_consts);
    XCAM_FAIL_RETURN (ERROR, xcam_ret_is_ok (ret), NULL, "vk-blend build VKBlendPyrShader failed");

    return worker;
}

static SmartPtr<VKWorker>
create_reconstruct_pyr_shader (const SmartPtr<VKBlender> &blender)
{
    SmartPtr<VKDevice> dev = blender->get_vk_device ();
    XCAM_ASSERT (dev.ptr ());

    ReconstructPushConstsProp prop;
    VKConstRange::VKPushConstArgs push_consts;
    push_consts.push_back (new VKReconstructPushConst (prop));

    VKDescriptor::BindingArray binding_layout;
    binding_layout.clear ();
    for (int i = 0; i < RECONSTRUCT_SHADER_BINDING_COUNT; ++i) {
        SmartPtr<VKDescriptor::SetLayoutBinding> binding =
            new VKDescriptor::ComputeLayoutBinding (VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, i);
        binding_layout.push_back (binding);
    }

    SmartPtr<VKWorker> worker = new VKWorker (dev, "VKReconstructShader", new CbReconstructPyr (blender));
    XCAM_ASSERT (worker.ptr ());

    XCamReturn ret = worker->build (shaders_info[ShaderReconstructPyr], binding_layout, push_consts);
    XCAM_FAIL_RETURN (ERROR, xcam_ret_is_ok (ret), NULL, "vk-blend build VKReconstructShader failed");

    return worker;
}

XCamReturn
BlenderImpl::create_workers (const SmartPtr<VKBlender> &blender)
{
    XCAM_ASSERT (blender.ptr ());

    VKWorkers::iterator i = _workers.find (ShaderGaussScalePyr);
    if (i == _workers.end ()) {
        SmartPtr<VKWorker> gauss_scale = create_gauss_scale_pyr_shader (blender);
        XCAM_ASSERT (gauss_scale.ptr ());
        _workers.insert (std::make_pair (ShaderGaussScalePyr, gauss_scale));
    }

    i = _workers.find (ShaderLapTransPyr);
    if (i == _workers.end ()) {
        SmartPtr<VKWorker> lap_trans = create_lap_trans_pyr_shader (blender);
        XCAM_ASSERT (lap_trans.ptr ());
        _workers.insert (std::make_pair (ShaderLapTransPyr, lap_trans));
    }

    i = _workers.find (ShaderBlendPyr);
    if (i == _workers.end ()) {
        SmartPtr<VKWorker> blend = create_blend_pyr_shader (blender);
        XCAM_ASSERT (blend.ptr ());
        _workers.insert (std::make_pair (ShaderBlendPyr, blend));
    }

    i = _workers.find (ShaderReconstructPyr);
    if (i == _workers.end ()) {
        SmartPtr<VKWorker> reconstruct = create_reconstruct_pyr_shader (blender);
        XCAM_ASSERT (reconstruct.ptr ());
        _workers.insert (std::make_pair (ShaderReconstructPyr, reconstruct));
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
BlenderImpl::redirect_workers ()
{
    VKWorkers::iterator i = _workers.find (ShaderGaussScalePyr);
    XCAM_ASSERT (i != _workers.end ());
    SmartPtr<VKWorker> gauss_scale = i->second;

    i = _workers.find (ShaderLapTransPyr);
    XCAM_ASSERT (i != _workers.end ());
    SmartPtr<VKWorker> lap_trans = i->second;

    i = _workers.find (ShaderBlendPyr);
    XCAM_ASSERT (i != _workers.end ());
    SmartPtr<VKWorker> top_blend = i->second;

    i = _workers.find (ShaderReconstructPyr);
    XCAM_ASSERT (i != _workers.end ());
    SmartPtr<VKWorker> reconstruct = i->second;

    XCAM_ASSERT (gauss_scale.ptr () && lap_trans.ptr () && reconstruct.ptr () && top_blend.ptr ());
    for (uint32_t i = 0; i < pyr_layers_num - 1; ++i) {
        PyrLayer &layer_next = pyr_layer[i + 1];
        layer_next.gauss_scale[VKBlender::BufIdx0] = gauss_scale.ptr ();
        layer_next.gauss_scale[VKBlender::BufIdx1] = gauss_scale.ptr ();

        PyrLayer &layer = pyr_layer[i];
        layer.lap_trans[VKBlender::BufIdx0] = lap_trans.ptr ();
        layer.lap_trans[VKBlender::BufIdx1] = lap_trans.ptr ();
        layer.reconstruct = reconstruct.ptr ();
    }

    PyrLayer &top_layer = pyr_layer[pyr_layers_num - 1];
    top_layer.blend = top_blend.ptr ();

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
BlenderImpl::fix_parameters ()
{
    for (uint32_t i = 0; i < pyr_layers_num - 1; ++i) {
        fix_gs_params (i + 1, VKBlender::BufIdx0);
        fix_gs_params (i + 1, VKBlender::BufIdx1);

        fix_lap_trans_params (i, VKBlender::BufIdx0);
        fix_lap_trans_params (i, VKBlender::BufIdx1);

        fix_reconstruct_params (i);
    }

    fix_blend_params ();

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
BlenderImpl::fix_gs_params (uint32_t level, VKBlender::BufIdx idx)
{
    XCAM_ASSERT (level >= 1);

    uint32_t level_in = level - 1;
    PyrLayer &layer_in = pyr_layer[level_in];
    PyrLayer &layer_out = pyr_layer[level];
    XCAM_ASSERT (layer_out.gs_buf[idx].ptr () && (layer_in.gs_buf[idx].ptr () || level == 1));

    VKDescriptor::BindingArray binding_layout;
    binding_layout.clear ();
    for (int i = 0; i < GS_SHADER_BINDING_COUNT; ++i) {
        SmartPtr<VKDescriptor::SetLayoutBinding> binding =
            new VKDescriptor::ComputeLayoutBinding (VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, i);
        binding_layout.push_back (binding);
    }

    VKDescriptor::SetBindInfoArray bindings (GS_SHADER_BINDING_COUNT);
    bindings[0].layout = binding_layout[0];
    bindings[1].layout = binding_layout[1];
    if (layer_in.gs_buf[idx].ptr ()) {
        bindings[0].desc = VKBufDesc (layer_in.gs_buf[idx], NV12PlaneYIdx);
        bindings[1].desc = VKBufDesc (layer_in.gs_buf[idx], NV12PlaneUVIdx);
    }
    bindings[2].layout = binding_layout[2];
    bindings[2].desc = VKBufDesc (layer_out.gs_buf[idx], NV12PlaneYIdx);
    bindings[3].layout = binding_layout[3];
    bindings[3].desc = VKBufDesc (layer_out.gs_buf[idx], NV12PlaneUVIdx);
    layer_out.gs_bindings[idx] = bindings;

    const VKBufInfo in_info = layer_in.gs_buf[idx]->get_buf_info ();
    const VKBufInfo out_info = layer_out.gs_buf[idx]->get_buf_info ();

    size_t unit_bytes = sizeof (uint32_t);
    GaussScalePushConstsProp prop;
    prop.in_img_width = XCAM_ALIGN_UP (in_info.width, unit_bytes) / unit_bytes;
    prop.in_img_height = in_info.height;
    prop.out_img_width = XCAM_ALIGN_UP (out_info.width, unit_bytes) / unit_bytes;
    prop.out_img_height = out_info.height;
    if (level == 1) {
        const Rect area = _blender->get_input_merge_area (idx);
        prop.in_offset_x = XCAM_ALIGN_UP (area.pos_x, unit_bytes) / unit_bytes;
        prop.merge_width = XCAM_ALIGN_UP (area.width, unit_bytes) / unit_bytes;
    } else {
        prop.in_offset_x = 0;
        prop.merge_width = XCAM_ALIGN_UP (in_info.width, unit_bytes) / unit_bytes;
    }
    layer_out.gs_consts[idx] = new VKGaussScalePushConst (prop);

    layer_out.gs_global_size[idx] = WorkSize (
        XCAM_ALIGN_UP (prop.out_img_width, 8) / 8,
        XCAM_ALIGN_UP (out_info.height, 16) / 16);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
BlenderImpl::fix_lap_trans_params (uint32_t level, VKBlender::BufIdx idx)
{
    XCAM_ASSERT (level < pyr_layers_num - 1);

    PyrLayer &layer = pyr_layer[level];
    PyrLayer &layer_next = pyr_layer[level + 1];
    XCAM_ASSERT ((layer.gs_buf[idx].ptr () || level == 0) && layer_next.gs_buf[idx].ptr ());
    XCAM_ASSERT (layer.lap_buf[idx].ptr ());

    VKDescriptor::BindingArray binding_layout;
    binding_layout.clear ();
    for (int i = 0; i < LAP_SHADER_BINDING_COUNT; ++i) {
        SmartPtr<VKDescriptor::SetLayoutBinding> binding =
            new VKDescriptor::ComputeLayoutBinding (VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, i);
        binding_layout.push_back (binding);
    }

    VKDescriptor::SetBindInfoArray bindings (LAP_SHADER_BINDING_COUNT);
    bindings[0].layout = binding_layout[0];
    bindings[1].layout = binding_layout[1];
    if (layer.gs_buf[idx].ptr ()) {
        bindings[0].desc = VKBufDesc (layer.gs_buf[idx], NV12PlaneYIdx);
        bindings[1].desc = VKBufDesc (layer.gs_buf[idx], NV12PlaneUVIdx);
    }
    bindings[2].layout = binding_layout[2];
    bindings[2].desc = VKBufDesc (layer_next.gs_buf[idx], NV12PlaneYIdx);
    bindings[3].layout = binding_layout[3];
    bindings[3].desc = VKBufDesc (layer_next.gs_buf[idx], NV12PlaneUVIdx);
    bindings[4].layout = binding_layout[4];
    bindings[4].desc = VKBufDesc (layer.lap_buf[idx], NV12PlaneYIdx);
    bindings[5].layout = binding_layout[5];
    bindings[5].desc = VKBufDesc (layer.lap_buf[idx], NV12PlaneUVIdx);
    layer.lap_bindings[idx] = bindings;

    const VKBufInfo in_info = layer.gs_buf[idx]->get_buf_info ();
    const VKBufInfo gs_info = layer_next.gs_buf[idx]->get_buf_info ();

    size_t unit_bytes = sizeof (uint32_t) * 2;
    LapPushConstsProp prop;
    prop.in_img_width = XCAM_ALIGN_UP (in_info.width, unit_bytes) / unit_bytes;
    prop.in_img_height = in_info.height;
    prop.gaussscale_img_width = XCAM_ALIGN_UP (gs_info.width, sizeof (uint32_t)) / sizeof (uint32_t);
    prop.gaussscale_img_height = gs_info.height;
    if (level == 0) {
        const Rect area = _blender->get_input_merge_area (idx);
        prop.in_offset_x = XCAM_ALIGN_UP (area.pos_x, unit_bytes) / unit_bytes;
        prop.merge_width = XCAM_ALIGN_UP (area.width, unit_bytes) / unit_bytes;
    } else {
        prop.in_offset_x = 0;
        prop.merge_width = XCAM_ALIGN_UP (in_info.width, unit_bytes) / unit_bytes;
    }
    layer.lap_consts[idx] = new VKLapPushConst (prop);

    layer.lap_global_size[idx] = WorkSize (
        XCAM_ALIGN_UP (prop.merge_width, 8) / 8,
        XCAM_ALIGN_UP (in_info.height, 32) / 32);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
BlenderImpl::fix_blend_params ()
{
    PyrLayer &top_layer = pyr_layer[pyr_layers_num - 1];
    XCAM_ASSERT (top_layer.gs_buf[VKBlender::BufIdx0].ptr () && top_layer.gs_buf[VKBlender::BufIdx1].ptr ());
    XCAM_ASSERT (top_layer.mask.ptr ());
    XCAM_ASSERT (top_layer.blend_buf.ptr ());

    VKDescriptor::BindingArray binding_layout;
    binding_layout.clear ();
    for (int i = 0; i < BLEND_SHADER_BINDING_COUNT; ++i) {
        SmartPtr<VKDescriptor::SetLayoutBinding> binding =
            new VKDescriptor::ComputeLayoutBinding (VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, i);
        binding_layout.push_back (binding);
    }

    VKDescriptor::SetBindInfoArray bindings (BLEND_SHADER_BINDING_COUNT);
    bindings[0].layout = binding_layout[0];
    bindings[0].desc = VKBufDesc (top_layer.gs_buf[VKBlender::BufIdx0], NV12PlaneYIdx);
    bindings[1].layout = binding_layout[1];
    bindings[1].desc = VKBufDesc (top_layer.gs_buf[VKBlender::BufIdx0], NV12PlaneUVIdx);
    bindings[2].layout = binding_layout[2];
    bindings[2].desc = VKBufDesc (top_layer.gs_buf[VKBlender::BufIdx1], NV12PlaneYIdx);
    bindings[3].layout = binding_layout[3];
    bindings[3].desc = VKBufDesc (top_layer.gs_buf[VKBlender::BufIdx1], NV12PlaneUVIdx);
    bindings[4].layout = binding_layout[4];
    bindings[4].desc = VKBufDesc (top_layer.blend_buf, NV12PlaneYIdx);
    bindings[5].layout = binding_layout[5];
    bindings[5].desc = VKBufDesc (top_layer.blend_buf, NV12PlaneUVIdx);
    bindings[6].layout = binding_layout[6];
    bindings[6].desc = VKBufDesc (top_layer.mask);
    top_layer.blend_bindings = bindings;

    const VKBufInfo in0_info = top_layer.gs_buf[VKBlender::BufIdx0]->get_buf_info ();
    size_t unit_bytes = sizeof (uint32_t) * 2;
    BlendPushConstsProp prop;
    prop.in_img_width = XCAM_ALIGN_UP (in0_info.width, unit_bytes) / unit_bytes;
    top_layer.blend_consts = new VKBlendPushConst (prop);

    top_layer.blend_global_size = WorkSize (
        XCAM_ALIGN_UP (prop.in_img_width, 8) / 8,
        XCAM_ALIGN_UP (in0_info.height, 16) / 16);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
BlenderImpl::fix_reconstruct_params (uint32_t level)
{
    XCAM_ASSERT (level < pyr_layers_num - 1);

    PyrLayer &layer = pyr_layer[level];
    PyrLayer &prev_layer = pyr_layer[level + 1];

    XCAM_ASSERT (layer.lap_buf[VKBlender::BufIdx0].ptr ());
    XCAM_ASSERT (layer.lap_buf[VKBlender::BufIdx1].ptr ());
    XCAM_ASSERT (prev_layer.reconstruct_buf.ptr () && (layer.reconstruct_buf.ptr () || level == 0));
    XCAM_ASSERT (layer.mask.ptr ());

    VKDescriptor::BindingArray binding_layout;
    binding_layout.clear ();
    for (int i = 0; i < RECONSTRUCT_SHADER_BINDING_COUNT; ++i) {
        SmartPtr<VKDescriptor::SetLayoutBinding> binding =
            new VKDescriptor::ComputeLayoutBinding (VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, i);
        binding_layout.push_back (binding);
    }

    VKDescriptor::SetBindInfoArray bindings (RECONSTRUCT_SHADER_BINDING_COUNT);
    bindings[0].layout = binding_layout[0];
    bindings[0].desc = VKBufDesc (layer.lap_buf[VKBlender::BufIdx0], NV12PlaneYIdx);
    bindings[1].layout = binding_layout[1];
    bindings[1].desc = VKBufDesc (layer.lap_buf[VKBlender::BufIdx0], NV12PlaneUVIdx);
    bindings[2].layout = binding_layout[2];
    bindings[2].desc = VKBufDesc (layer.lap_buf[VKBlender::BufIdx1], NV12PlaneYIdx);
    bindings[3].layout = binding_layout[3];
    bindings[3].desc = VKBufDesc (layer.lap_buf[VKBlender::BufIdx1], NV12PlaneUVIdx);
    bindings[4].layout = binding_layout[4];
    bindings[5].layout = binding_layout[5];
    if (layer.reconstruct_buf.ptr ()) {
        bindings[4].desc = VKBufDesc (layer.reconstruct_buf, NV12PlaneYIdx);
        bindings[5].desc = VKBufDesc (layer.reconstruct_buf, NV12PlaneUVIdx);
    }
    bindings[6].layout = binding_layout[6];
    bindings[6].desc = VKBufDesc (prev_layer.reconstruct_buf, NV12PlaneYIdx);
    bindings[7].layout = binding_layout[7];
    bindings[7].desc = VKBufDesc (prev_layer.reconstruct_buf, NV12PlaneUVIdx);
    bindings[8].layout = binding_layout[8];
    bindings[8].desc = VKBufDesc (layer.mask);
    layer.reconstruct_bindings = bindings;

    const VKBufInfo lap0_info = layer.lap_buf[VKBlender::BufIdx0]->get_buf_info ();
    const VKBufInfo prev_recons_info = prev_layer.reconstruct_buf->get_buf_info ();

    size_t unit_bytes = sizeof (uint32_t) * 2;
    ReconstructPushConstsProp prop;
    prop.lap_img_width = XCAM_ALIGN_UP (lap0_info.width, unit_bytes) / unit_bytes;
    prop.lap_img_height = lap0_info.height;
    prop.prev_blend_img_width = XCAM_ALIGN_UP (prev_recons_info.width, sizeof (uint32_t)) / sizeof (uint32_t);
    prop.prev_blend_img_height = prev_recons_info.height;
    if (level == 0) {
        const VideoBufferInfo info = _blender->get_out_video_info ();
        prop.out_img_width = XCAM_ALIGN_UP (info.width, unit_bytes) / unit_bytes;

        const Rect area = _blender->get_merge_window ();
        prop.out_offset_x = XCAM_ALIGN_UP (area.pos_x, unit_bytes) / unit_bytes;
    } else {
        const VKBufInfo info = layer.reconstruct_buf->get_buf_info ();
        prop.out_img_width = XCAM_ALIGN_UP (info.width, unit_bytes) / unit_bytes;
        prop.out_offset_x = 0;
    }
    layer.reconstruct_consts = new VKReconstructPushConst (prop);

    layer.reconstruct_global_size = WorkSize (
        XCAM_ALIGN_UP (prop.lap_img_width, 8) / 8,
        XCAM_ALIGN_UP (lap0_info.height, 32) / 32);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
BlenderImpl::start_gauss_scale (uint32_t level, VKBlender::BufIdx idx)
{
    XCAM_ASSERT (level >= 1 && level < pyr_layers_num);

    PyrLayer &layer = pyr_layer[level];
    layer.gauss_scale[idx]->set_global_size (layer.gs_global_size[idx]);

    SmartPtr<BlendArgs> args = new BlendArgs (level, idx);
    args->set_bindings (layer.gs_bindings[idx]);
    args->add_push_const (layer.gs_consts[idx]);

    return layer.gauss_scale[idx]->work (args);
}

XCamReturn
BlenderImpl::start_lap_tran (uint32_t level, VKBlender::BufIdx idx)
{
    PyrLayer &layer = pyr_layer[level];

    SmartPtr<VKBlender::Sync> &sync = layer.lap_sync[idx];
    if (!sync->is_synced ())
        return XCAM_RETURN_NO_ERROR;
    sync->reset ();

    layer.lap_trans[idx]->set_global_size (layer.lap_global_size[idx]);

    SmartPtr<BlendArgs> args = new BlendArgs (level, idx);
    args->set_bindings (layer.lap_bindings[idx]);
    args->add_push_const (layer.lap_consts[idx]);

    return layer.lap_trans[idx]->work (args);
}

XCamReturn
BlenderImpl::start_lap_trans (uint32_t level, VKBlender::BufIdx idx)
{
    XCAM_ASSERT (level >= 1 && level < pyr_layers_num);

    uint32_t pre_level = level - 1;
    pyr_layer[pre_level].lap_sync[idx]->increment ();

    XCamReturn ret = start_lap_tran (pre_level, idx);
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret,
        "vk-blend start lap tran failed, level:%d idx:%d", pre_level, idx);

    if (level == pyr_layers_num - 1)
        return XCAM_RETURN_NO_ERROR;
    pyr_layer[level].lap_sync[idx]->increment ();

    ret = start_lap_tran (level, idx);
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret,
        "vk-blend start lap tran failed, level:%d idx:%d", level, idx);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
BlenderImpl::start_top_blend ()
{
    uint32_t level = pyr_layers_num - 1;
    PyrLayer &layer = pyr_layer[level];

    SmartPtr<VKBlender::Sync> &sync = layer.blend_sync;
    if (!sync->is_synced ())
        return XCAM_RETURN_NO_ERROR;
    sync->reset ();

    SmartPtr<VKWorker::VKArguments> args = new VKWorker::VKArguments;
    args->set_bindings (layer.blend_bindings);
    args->add_push_const (layer.blend_consts);

    layer.blend->set_global_size (layer.blend_global_size);

    return layer.blend->work (args);
}

XCamReturn
BlenderImpl::start_reconstruct (uint32_t level)
{
    XCAM_ASSERT (level < pyr_layers_num - 1);
    PyrLayer &layer = pyr_layer[level];

    SmartPtr<VKBlender::Sync> &sync = layer.reconstruct_sync;
    if (!sync->is_synced ())
        return XCAM_RETURN_NO_ERROR;
    sync->reset ();

    SmartPtr<BlendArgs> args = new BlendArgs (level);
    args->set_bindings (layer.reconstruct_bindings);
    args->add_push_const (layer.reconstruct_consts);

    layer.reconstruct->set_global_size (layer.reconstruct_global_size);

    return layer.reconstruct->work (args);
}

XCamReturn
BlenderImpl::stop ()
{
    for (uint32_t lv = 0; lv < pyr_layers_num; ++lv) {
        pyr_layer[lv].gs_buf[VKBlender::BufIdx0].release ();
        pyr_layer[lv].gs_buf[VKBlender::BufIdx1].release ();
        pyr_layer[lv].lap_buf[VKBlender::BufIdx0].release ();
        pyr_layer[lv].lap_buf[VKBlender::BufIdx1].release ();
        pyr_layer[lv].reconstruct_buf.release ();
        pyr_layer[lv].blend_buf.release ();

        pyr_layer[lv].gs_consts[VKBlender::BufIdx0].release ();
        pyr_layer[lv].gs_consts[VKBlender::BufIdx1].release ();
        pyr_layer[lv].lap_consts[VKBlender::BufIdx0].release ();
        pyr_layer[lv].lap_consts[VKBlender::BufIdx1].release ();
        pyr_layer[lv].reconstruct_consts.release ();
        pyr_layer[lv].blend_consts.release ();
    }

    return XCAM_RETURN_NO_ERROR;
}

}

VKBlender::VKBlender (const SmartPtr<VKDevice> dev, const char *name)
    : VKHandler (dev, name)
    , Blender (VK_BLENDER_ALIGN_X, VK_BLENDER_ALIGN_Y)
{
    SmartPtr<VKBlenderPriv::BlenderImpl> impl =
        new VKBlenderPriv::BlenderImpl (this, XCAM_VK_DEFAULT_LEVEL);
    XCAM_ASSERT (impl.ptr ());

    _impl = impl;
}

VKBlender::~VKBlender ()
{
}

XCamReturn
VKBlender::blend (
    const SmartPtr<VideoBuffer> &in0, const SmartPtr<VideoBuffer> &in1, SmartPtr<VideoBuffer> &out_buf)
{
    XCAM_ASSERT (in0.ptr () && in1.ptr ());

    SmartPtr<BlenderParam> param = new BlenderParam (in0, in1, out_buf);
    XCAM_ASSERT (param.ptr ());

    XCamReturn ret = execute_buffer (param, true);
    XCAM_FAIL_RETURN (ERROR, xcam_ret_is_ok (ret), ret, "vk-blend execute buffer failed");

    finish ();
    if (!out_buf.ptr ()) {
        out_buf = param->out_buf;
    }

    return ret;
}

static XCamReturn
check_merge_area (const SmartPtr<VKBlender> &blender)
{
    Rect in0_area, in1_area, out_area;

    in0_area = blender->get_input_merge_area (VKBlender::BufIdx0);
    XCAM_FAIL_RETURN (
        ERROR,
        in0_area.pos_y == 0 && in0_area.width && in0_area.height &&
        in0_area.pos_x % VK_BLENDER_ALIGN_X == 0 &&
        in0_area.width % VK_BLENDER_ALIGN_X == 0 &&
        in0_area.height % VK_BLENDER_ALIGN_Y == 0,
        XCAM_RETURN_ERROR_PARAM,
        "vk-blend invalid input0 merge area, pos_x:%d, pos_y:%d, width:%d, height:%d",
        in0_area.pos_x, in0_area.pos_y, in0_area.width, in0_area.height);

    in1_area = blender->get_input_merge_area (VKBlender::BufIdx1);
    XCAM_FAIL_RETURN (
        ERROR,
        in1_area.pos_y == 0 && in1_area.width && in1_area.height &&
        in1_area.pos_x % VK_BLENDER_ALIGN_X == 0 &&
        in1_area.width % VK_BLENDER_ALIGN_X == 0 &&
        in1_area.height % VK_BLENDER_ALIGN_Y == 0,
        XCAM_RETURN_ERROR_PARAM,
        "vk-blend invalid input1 merge area, pos_x:%d, pos_y:%d, width:%d, height:%d",
        in1_area.pos_x, in1_area.pos_y, in1_area.width, in1_area.height);

    out_area = blender->get_merge_window ();
    XCAM_FAIL_RETURN (
        ERROR,
        out_area.pos_y == 0 && out_area.width && out_area.height &&
        out_area.pos_x % VK_BLENDER_ALIGN_X == 0 &&
        out_area.width % VK_BLENDER_ALIGN_X == 0 &&
        out_area.height % VK_BLENDER_ALIGN_Y == 0,
        XCAM_RETURN_ERROR_PARAM,
        "vk-blend invalid output merge area, pos_x:%d, pos_y:%d, width:%d, height:%d",
        out_area.pos_x, out_area.pos_y, out_area.width, out_area.height);

    XCAM_FAIL_RETURN (
        ERROR,
        in0_area.width == in1_area.width && in0_area.height == in1_area.height &&
        in0_area.width == out_area.width && in0_area.height == out_area.height,
        XCAM_RETURN_ERROR_PARAM,
        "vk-blend invalid input or output overlap area, input0:%dx%d, input1:%dx%d, output:%dx%d",
        in0_area.width, in0_area.height, in1_area.width, in1_area.height, out_area.width, out_area.height);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
VKBlender::set_output_info (const SmartPtr<ImageHandler::Parameters> &param)
{
    const VideoBufferInfo &in0_info = param->in_buf->get_video_info ();
    XCAM_FAIL_RETURN (
        ERROR, in0_info.format == V4L2_PIX_FMT_NV12, XCAM_RETURN_ERROR_PARAM,
        "vk-blend only support NV12 format, but input format is %s",
        xcam_fourcc_to_string (in0_info.format));

    uint32_t out_width, out_height;
    get_output_size (out_width, out_height);
    XCAM_FAIL_RETURN (
        ERROR, out_width && out_height, XCAM_RETURN_ERROR_PARAM,
        "vk-blend invalid output size:%dx%d", out_width, out_height);

    VideoBufferInfo out_info;
    out_info.init (
        in0_info.format, out_width, out_height,
        XCAM_ALIGN_UP (out_width, VK_BLENDER_ALIGN_X),
        XCAM_ALIGN_UP (out_height, VK_BLENDER_ALIGN_Y));
    set_out_video_info (out_info);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
VKBlender::configure_resource (const SmartPtr<Parameters> &param)
{
    XCAM_ASSERT (param.ptr ());
    XCAM_ASSERT (_impl->pyr_layers_num <= XCAM_VK_MAX_LEVEL);

    SmartPtr<BlenderParam> blend_param = param.dynamic_cast_ptr<BlenderParam> ();
    XCAM_ASSERT (blend_param.ptr () && blend_param->in_buf.ptr () && blend_param->in1_buf.ptr ());

    XCamReturn ret = set_output_info (param);
    XCAM_FAIL_RETURN (ERROR, xcam_ret_is_ok (ret), ret, "vk-blend set output video info failed");

    ret = check_merge_area (this);
    XCAM_FAIL_RETURN (ERROR, xcam_ret_is_ok (ret), ret, "vk-blend check merge area failed");

    _impl->init_syncs ();

    ret = _impl->init_layers_bufs (param);
    XCAM_FAIL_RETURN (ERROR, xcam_ret_is_ok (ret), ret, "vk-blend init buffers failed");

    ret = _impl->fix_parameters ();
    XCAM_FAIL_RETURN (ERROR, xcam_ret_is_ok (ret), ret, "vk-blend fix parameters failed");

    ret = _impl->create_workers (this);
    XCAM_FAIL_RETURN (ERROR, xcam_ret_is_ok (ret), ret, "vk-blend create workers failed");

    ret = _impl->redirect_workers ();
    XCAM_FAIL_RETURN (ERROR, xcam_ret_is_ok (ret), ret, "vk-blend redirect workers failed");

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
VKBlender::start_work (const SmartPtr<ImageHandler::Parameters> &param)
{
    SmartPtr<VKBlender::BlenderParam> blend_param = param.dynamic_cast_ptr<BlenderParam> ();
    XCAM_ASSERT (blend_param.ptr ());
    XCAM_ASSERT (blend_param->in_buf.ptr () && blend_param->in1_buf.ptr () && blend_param->out_buf.ptr ());

#if DUMP_BUFFER
    SmartPtr<VKVideoBuffer> in0_vk = blend_param->in_buf.dynamic_cast_ptr<VKVideoBuffer> ();
    SmartPtr<VKVideoBuffer> in1_vk = blend_param->in1_buf.dynamic_cast_ptr<VKVideoBuffer> ();
    XCAM_ASSERT (in0_vk.ptr () && in1_vk.ptr ());
    dump_level_vkbuf (in0_vk->get_vk_buf (), "blend-in", 0, VKBlender::BufIdx0);
    dump_level_vkbuf (in1_vk->get_vk_buf (), "blend-in", 0, VKBlender::BufIdx1);
#endif

    _impl->bind_io_bufs_to_layer0 (blend_param->in_buf, blend_param->in1_buf, blend_param->out_buf);
    _impl->bind_io_vkbufs_to_desc ();

    _impl->pyr_layer[0].lap_sync[BufIdx0]->increment ();
    _impl->pyr_layer[0].lap_sync[BufIdx1]->increment ();

    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    ret = _impl->start_gauss_scale (1, BufIdx0);
    CHECK_RET (ret, "vk-blend start gauss scale failed, level:1 index:0\n");

    ret = _impl->start_gauss_scale (1, BufIdx1);
    CHECK_RET (ret, "vk-blend start gauss scale failed, level:1 index:1\n");

    execute_done (param, ret);

    return ret;
}

void
VKBlender::gauss_scale_done (
    const SmartPtr<Worker> &worker, const SmartPtr<Worker::Arguments> &base, const XCamReturn error)
{
    if (!xcam_ret_is_ok (error)) {
        XCAM_LOG_ERROR ("vk-blend gauss scale failed");
        return ;
    }

    SmartPtr<VKBlenderPriv::BlendArgs> args = base.dynamic_cast_ptr<VKBlenderPriv::BlendArgs> ();
    XCAM_ASSERT (args.ptr ());
    uint32_t level = args->get_level ();
    uint32_t next_level = level + 1;
    BufIdx idx = args->get_idx ();

    SmartPtr<VKWorker> gs_worker = worker.dynamic_cast_ptr<VKWorker> ();
    XCAM_ASSERT (gs_worker.ptr ());
    gs_worker->wait_fence ();

#if DUMP_BUFFER
    dump_level_vkbuf (_impl->pyr_layer[level].gs_buf[idx], "gauss-scale", level, idx);
#endif

    XCamReturn ret = _impl->start_lap_trans (level, idx);
    CHECK_RET (ret, "vk-blend execute laplace transformation failed, level:%d idx:%d", level, idx);

    if (next_level == _impl->pyr_layers_num) {
        _impl->pyr_layer[level].blend_sync->increment ();

        ret = _impl->start_top_blend ();
        CHECK_RET (ret, "vk-blend execute top blend failed, level:%d idx:%d", level, idx);
    } else {
        ret = _impl->start_gauss_scale (next_level, idx);
        CHECK_RET (ret, "vk-blend execute gauss scale failed, level:%d idx:%d", next_level, idx);
    }
}

void
VKBlender::lap_trans_done (
    const SmartPtr<Worker> &worker, const SmartPtr<Worker::Arguments> &base, const XCamReturn error)
{
    XCAM_UNUSED (base);
    if (!xcam_ret_is_ok (error)) {
        XCAM_LOG_ERROR ("vk-blend laplace transformation failed");
        return ;
    }

    SmartPtr<VKBlenderPriv::BlendArgs> args = base.dynamic_cast_ptr<VKBlenderPriv::BlendArgs> ();
    XCAM_ASSERT (args.ptr ());
    uint32_t level = args->get_level ();

    SmartPtr<VKWorker> laptrans_worker = worker.dynamic_cast_ptr<VKWorker> ();
    XCAM_ASSERT (laptrans_worker.ptr ());
    laptrans_worker->wait_fence ();

#if DUMP_BUFFER
    BufIdx idx = args->get_idx ();
    dump_level_vkbuf (_impl->pyr_layer[level].lap_buf[idx], "lap", level, idx);
#endif

    _impl->pyr_layer[level].reconstruct_sync->increment ();

    XCamReturn ret = _impl->start_reconstruct (level);
    CHECK_RET (ret, "vk-blend execute reconstruct failed, level:%d", level);
}

void
VKBlender::blend_done (
    const SmartPtr<Worker> &worker, const SmartPtr<Worker::Arguments> &base, const XCamReturn error)
{
    XCAM_UNUSED (base);
    if (!xcam_ret_is_ok (error)) {
        XCAM_LOG_ERROR ("vk-blend blend failed");
        return ;
    }

    SmartPtr<VKWorker> blend_worker = worker.dynamic_cast_ptr<VKWorker> ();
    XCAM_ASSERT (blend_worker.ptr ());
    blend_worker->wait_fence ();

#if DUMP_BUFFER
    dump_vkbuf (_impl->pyr_layer[_impl->pyr_layers_num - 1].blend_buf, "blend-top");
#endif

    uint32_t pre_level = _impl->pyr_layers_num - 2;
    _impl->pyr_layer[pre_level].reconstruct_sync->increment ();

    XCamReturn ret = _impl->start_reconstruct (pre_level);
    CHECK_RET (ret, "vk-blend execute reconstruct failed, level:%d", pre_level);
}

void
VKBlender::reconstruct_done (
    const SmartPtr<Worker> &worker, const SmartPtr<Worker::Arguments> &base, const XCamReturn error)
{
    XCAM_UNUSED (base);
    if (!xcam_ret_is_ok (error)) {
        XCAM_LOG_ERROR ("vk-blend reconstruct failed");
        return ;
    }

    SmartPtr<VKBlenderPriv::BlendArgs> args = base.dynamic_cast_ptr<VKBlenderPriv::BlendArgs> ();
    XCAM_ASSERT (args.ptr ());
    uint32_t level = args->get_level ();

    SmartPtr<VKWorker> reconstruct_worker = worker.dynamic_cast_ptr<VKWorker> ();
    XCAM_ASSERT (reconstruct_worker.ptr ());
    reconstruct_worker->wait_fence ();

#if DUMP_BUFFER
    BufIdx idx = args->get_idx ();
    dump_level_vkbuf (_impl->pyr_layer[level].reconstruct_buf, "reconstruct", level, idx);
#endif

    if (level == 0) {
        return;
    }

    uint32_t pre_level = level - 1;
    _impl->pyr_layer[pre_level].reconstruct_sync->increment ();

    XCamReturn ret = _impl->start_reconstruct (pre_level);
    CHECK_RET (ret, "vk-blend execute reconstruct failed, level:%d", pre_level);
}

SmartPtr<VKHandler>
create_vk_blender (const SmartPtr<VKDevice> &dev)
{
    SmartPtr<VKBlender> blender = new VKBlender (dev);
    XCAM_ASSERT (blender.ptr ());

    return blender;
}

SmartPtr<Blender>
Blender::create_vk_blender (const SmartPtr<VKDevice> &dev)
{
    SmartPtr<VKHandler> handler = XCam::create_vk_blender (dev);
    return handler.dynamic_cast_ptr<Blender> ();
}

}
