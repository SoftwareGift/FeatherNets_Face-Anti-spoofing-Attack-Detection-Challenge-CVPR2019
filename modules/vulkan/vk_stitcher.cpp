/*
 * vk_stitcher.cpp - Vulkan stitcher implementation
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

#include "surview_fisheye_dewarp.h"
#include "vk_video_buf_allocator.h"
#include "vk_geomap_handler.h"
#include "vk_blender.h"
#include "vk_copy_handler.h"
#include "vk_stitcher.h"

#define DUMP_BUFFER 0

#define GEOMAP_POOL_SIZE 1

#define VK_STITCHER_ALIGNMENT_X 16
#define VK_STITCHER_ALIGNMENT_Y 4

#define MAP_FACTOR_X 16
#define MAP_FACTOR_Y 16

#define CHECK_RET(ret, format, ...) \
    if (!xcam_ret_is_ok (ret)) { \
        XCAM_LOG_ERROR (format, ## __VA_ARGS__); \
    }

namespace XCam {

#if DUMP_BUFFER
static void
dump_buf (const SmartPtr<VideoBuffer> &buf, uint32_t idx, const char *prefix)
{
    XCAM_ASSERT (buf.ptr () && prefix);

    char name[256];
    snprintf (name, 256, "%s-%d", prefix, idx);
    dump_buf_perfix_path (buf, name);
}
#endif

namespace VKSitcherPriv {

DECLARE_HANDLER_CALLBACK (CbGeoMap, VKStitcher, geomap_done);

struct GeoMapParam
    : ImageHandler::Parameters
{
    SmartPtr<VKStitcher::StitcherParam>    stitch_param;
    uint32_t                               idx;

    GeoMapParam (uint32_t i)
        : idx (i)
    {}
};

struct Factor {
    float x, y;

    Factor () : x (1.0f), y (1.0f) {}
    void reset () {
        x = 1.0f;
        y = 1.0f;
    }
};

typedef std::vector<SmartPtr<VKCopyHandler>> Copiers;

struct StitcherResource {
    SmartPtr<VKBlender::Sync>             blender_sync[XCAM_STITCH_MAX_CAMERAS];
    SmartPtr<BufferPool>                  mapper_pool[XCAM_STITCH_MAX_CAMERAS];

    SmartPtr<GeoMapParam>                 mapper_param[XCAM_STITCH_MAX_CAMERAS];
    SmartPtr<VKBlender::BlenderParam>     blender_param[XCAM_STITCH_MAX_CAMERAS];
    SmartPtr<ImageHandler::Parameters>    copier_param[XCAM_STITCH_MAX_CAMERAS];

    SmartPtr<VKGeoMapHandler>             mapper[XCAM_STITCH_MAX_CAMERAS];
    SmartPtr<VKBlender>                   blender[XCAM_STITCH_MAX_CAMERAS];
    Copiers                               copiers;

    StitcherResource ();
};

class StitcherImpl {
    friend class XCam::VKStitcher;

public:
    StitcherImpl (VKStitcher *handler)
        : _stitcher (handler)
    {}

    XCamReturn init_resource ();

    XCamReturn start_geo_mappers (const SmartPtr<VKStitcher::StitcherParam> &param);
    XCamReturn start_blenders (const SmartPtr<VKStitcher::StitcherParam> &param, uint32_t idx);
    XCamReturn start_copier (const SmartPtr<VKStitcher::StitcherParam> &param, uint32_t idx);

    XCamReturn stop ();

private:
    SmartPtr<VKGeoMapHandler> create_geo_mapper (
        const SmartPtr<VKDevice> &dev, const Stitcher::RoundViewSlice &view_slice);

    XCamReturn init_geo_mappers (const SmartPtr<VKDevice> &dev);
    XCamReturn init_blenders (const SmartPtr<VKDevice> &dev);
    XCamReturn init_copiers (const SmartPtr<VKDevice> &dev);

    bool update_geomap_factors (uint32_t idx);
    XCamReturn create_geomap_pool (const SmartPtr<VKDevice> &dev, uint32_t idx);
    XCamReturn set_geomap_table (
        const SmartPtr<VKGeoMapHandler> &mapper, const CameraInfo &cam_info,
        const Stitcher::RoundViewSlice &view_slice, const BowlDataConfig &bowl);
    XCamReturn generate_geomap_table (const SmartPtr<VKGeoMapHandler> &mapper, uint32_t idx);

    void update_blender_sync (uint32_t idx);
    XCamReturn start_blender (const SmartPtr<VKStitcher::StitcherParam> &param, uint32_t idx);

private:
    StitcherResource              _res;
    VKStitcher                   *_stitcher;
};

StitcherResource::StitcherResource ()
{
}

SmartPtr<VKGeoMapHandler>
StitcherImpl::create_geo_mapper (
    const SmartPtr<VKDevice> &dev, const Stitcher::RoundViewSlice &view_slice)
{
    XCAM_UNUSED (view_slice);

    SmartPtr<VKGeoMapHandler> mapper;
    if (_stitcher->get_scale_mode () == ScaleSingleConst) {
        mapper = new VKGeoMapHandler (dev, "sitcher_singleconst_remapper");
    } else {
        XCAM_LOG_ERROR (
            "vk-stitcher(%s) unsupported scale mode:%d",
            XCAM_STR (_stitcher->get_name ()), _stitcher->get_scale_mode ());
    }
    XCAM_ASSERT (mapper.ptr ());

    return mapper;
}

void
StitcherImpl::update_blender_sync (uint32_t idx)
{
    uint32_t cam_num = _stitcher->get_camera_num ();
    uint32_t pre_idx = (idx + cam_num - 1) % cam_num;

    _res.blender_sync[pre_idx]->increment ();
    _res.blender_sync[idx]->increment ();
}

bool
StitcherImpl::update_geomap_factors (uint32_t idx)
{
    SmartPtr<VKGeoMapHandler> &mapper = _res.mapper[idx];
    XCAM_FAIL_RETURN (
        ERROR, mapper.ptr (), false,
        "vk-stitcher(%s) geomap handler is empty, idx:%d", XCAM_STR (_stitcher->get_name ()), idx);

    if (_stitcher->get_scale_mode () == ScaleSingleConst) {
        Factor unify_factor, last_left_factor, last_right_factor;

        mapper->get_factors (unify_factor.x, unify_factor.y);
        if (XCAM_DOUBLE_EQUAL_AROUND (unify_factor.x, 0.0f) ||
                XCAM_DOUBLE_EQUAL_AROUND (unify_factor.y, 0.0f)) { // not started.
            return true;
        }

        last_left_factor = last_right_factor = unify_factor;
        unify_factor.x = (last_left_factor.x + last_right_factor.x) / 2.0f;
        unify_factor.y = (last_left_factor.y + last_right_factor.y) / 2.0f;

        mapper->set_factors (unify_factor.x, unify_factor.y);
    } else {
        XCAM_LOG_ERROR (
            "vk-stitcher(%s) unsupported scale mode:%d",
            XCAM_STR (_stitcher->get_name ()), _stitcher->get_scale_mode ());
        return false;
    }

    return true;
}

XCamReturn
StitcherImpl::create_geomap_pool (const SmartPtr<VKDevice> &dev, uint32_t idx)
{
    uint32_t output_width, output_height;
    _res.mapper[idx]->get_output_size (output_width, output_height);

    VideoBufferInfo out_info;
    out_info.init (
        V4L2_PIX_FMT_NV12, output_width, output_height,
        XCAM_ALIGN_UP (output_width, VK_STITCHER_ALIGNMENT_X),
        XCAM_ALIGN_UP (output_height, VK_STITCHER_ALIGNMENT_Y));

    SmartPtr<BufferPool> pool = create_vk_buffer_pool (dev);
    XCAM_ASSERT (pool.ptr ());
    XCAM_FAIL_RETURN (
        ERROR, pool->set_video_info (out_info) && pool->reserve (GEOMAP_POOL_SIZE),
        XCAM_RETURN_ERROR_MEM,
        "vk-stitcher(%s) create buffer pool failed, buffer size:%dx%d, idx:%d",
        XCAM_STR (_stitcher->get_name ()), out_info.width, out_info.height, idx);

    _res.mapper_pool[idx] = pool;

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
StitcherImpl::set_geomap_table (
    const SmartPtr<VKGeoMapHandler> &mapper, const CameraInfo &cam_info,
    const Stitcher::RoundViewSlice &view_slice, const BowlDataConfig &bowl)
{
    PolyFisheyeDewarp fd;
    fd.set_intrinsic_param (cam_info.calibration.intrinsic);
    fd.set_extrinsic_param (cam_info.calibration.extrinsic);

    uint32_t table_width, table_height;
    table_width = view_slice.width / MAP_FACTOR_X;
    table_height = view_slice.height / MAP_FACTOR_Y;

    SurViewFisheyeDewarp::MapTable map_table (table_width * table_height);
    fd.fisheye_dewarp (
        map_table, table_width, table_height,
        view_slice.width, view_slice.height, bowl);

    bool ret = mapper->set_lookup_table (map_table.data (), table_width, table_height);
    XCAM_FAIL_RETURN (
        ERROR, ret, XCAM_RETURN_ERROR_UNKNOWN,
        "vk-stitcher(%s)  set geomap lookup table failed", XCAM_STR (_stitcher->get_name ()));

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
StitcherImpl::generate_geomap_table (
    const SmartPtr<VKGeoMapHandler> &mapper, uint32_t idx)
{
    CameraInfo cam_info;
    _stitcher->get_camera_info (idx, cam_info);
    Stitcher::RoundViewSlice view_slice = _stitcher->get_round_view_slice (idx);

    BowlDataConfig bowl = _stitcher->get_bowl_config ();
    bowl.angle_start = view_slice.hori_angle_start;
    bowl.angle_end = format_angle (view_slice.hori_angle_start + view_slice.hori_angle_range);
    if (bowl.angle_end < bowl.angle_start)
        bowl.angle_start -= 360.0f;

    XCAM_LOG_DEBUG (
        "vk-stitcher(%s) camera(idx:%d) info(angle start:%.2f, range:%.2f), bowl info(angle start:%.2f, end:%.2f)",
        XCAM_STR (_stitcher->get_name ()), idx,
        view_slice.hori_angle_start, view_slice.hori_angle_range,
        bowl.angle_start, bowl.angle_end);

    XCamReturn ret = set_geomap_table (mapper, cam_info, view_slice, bowl);
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret,
        "vk-stitcher(%s) set geometry map table failed, idx:%d", XCAM_STR (_stitcher->get_name ()), idx);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
StitcherImpl::init_geo_mappers (const SmartPtr<VKDevice> &dev)
{
    uint32_t cam_num = _stitcher->get_camera_num ();
    SmartPtr<ImageHandler::Callback> cb = new CbGeoMap (_stitcher);

    for (uint32_t idx = 0; idx < cam_num; ++idx) {
        Stitcher::RoundViewSlice view_slice = _stitcher->get_round_view_slice (idx);

        SmartPtr<VKGeoMapHandler> &mapper = _res.mapper[idx];
        mapper = create_geo_mapper (dev, view_slice);
        mapper->set_callback (cb);
        mapper->set_output_size (view_slice.width, view_slice.height);

        create_geomap_pool (dev, idx);

        SmartPtr<GeoMapParam> &mapper_param = _res.mapper_param[idx];
        mapper_param = new GeoMapParam (idx);
        XCAM_ASSERT (mapper_param.ptr ());

        mapper_param->out_buf = _res.mapper_pool[idx]->get_buffer ();
        XCAM_ASSERT (mapper_param->out_buf.ptr ());

        XCamReturn ret = generate_geomap_table (mapper, idx);
        XCAM_FAIL_RETURN (
            ERROR, xcam_ret_is_ok (ret), ret,
            "vk-stitcher(%s) generate geomap table failed", XCAM_STR (_stitcher->get_name ()));
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
StitcherImpl::init_blenders (const SmartPtr<VKDevice> &dev)
{
    uint32_t out_width, out_height;
    _stitcher->get_output_size (out_width, out_height);
    uint32_t cam_num = _stitcher->get_camera_num ();

    for (uint32_t idx = 0; idx < cam_num; ++idx) {
        SmartPtr<VKBlender> &blender = _res.blender[idx];
        blender = create_vk_blender (dev).dynamic_cast_ptr<VKBlender> ();
        XCAM_ASSERT (blender.ptr ());

        const Stitcher::ImageOverlapInfo &overlap = _stitcher->get_overlap (idx);
        blender->set_output_size (out_width, out_height);
        blender->set_merge_window (overlap.out_area);
        blender->set_input_valid_area (overlap.left, 0);
        blender->set_input_valid_area (overlap.right, 1);
        blender->set_input_merge_area (overlap.left, 0);
        blender->set_input_merge_area (overlap.right, 1);

        SmartPtr<VKBlender::BlenderParam> &blender_param = _res.blender_param[idx];
        blender_param = new VKBlender::BlenderParam (NULL, NULL, NULL);
        XCAM_ASSERT (blender_param.ptr ());

        uint32_t next_idx = (idx + 1) % cam_num;
        blender_param->in_buf = _res.mapper_param[idx]->out_buf;
        blender_param->in1_buf = _res.mapper_param[next_idx]->out_buf;
        XCAM_ASSERT (blender_param->in_buf.ptr () && blender_param->in1_buf.ptr ());

        _res.blender_sync[idx] = new VKBlender::Sync (2);
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
StitcherImpl::init_copiers (const SmartPtr<VKDevice> &dev)
{
    uint32_t cam_num = _stitcher->get_camera_num ();
    for (uint32_t idx = 0; idx < cam_num; ++idx) {
        SmartPtr<ImageHandler::Parameters> &copier_param = _res.copier_param[idx];
        copier_param = new ImageHandler::Parameters ();
        XCAM_ASSERT (copier_param.ptr ());

        copier_param->in_buf = _res.mapper_param[idx]->out_buf;
        XCAM_ASSERT (copier_param->in_buf.ptr ());
    }

    Stitcher::CopyAreaArray areas = _stitcher->get_copy_area ();
    uint32_t size = areas.size ();
    for (uint32_t idx = 0; idx < size; ++idx) {
        XCAM_ASSERT (areas[idx].in_idx < size);

        SmartPtr<VKCopyHandler> copier = new VKCopyHandler (dev);
        XCAM_ASSERT (copier.ptr ());
        copier->enable_allocator (false);
        copier->set_copy_area (areas[idx].in_idx, areas[idx].in_area, areas[idx].out_area);

        _res.copiers.push_back (copier);
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
StitcherImpl::init_resource ()
{
    const SmartPtr<VKDevice> &dev = _stitcher->get_vk_device ();
    XCAM_ASSERT (dev.ptr ());

    XCamReturn ret = init_geo_mappers (dev);
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret,
        "vk-stitcher(%s) init dewarps failed", XCAM_STR (_stitcher->get_name ()));

    ret = init_blenders (dev);
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret,
        "vk-stitcher(%s) init blenders failed", XCAM_STR (_stitcher->get_name ()));

    ret = init_copiers (dev);
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret,
        "vk-stitcher(%s) init copiers failed", XCAM_STR (_stitcher->get_name ()));

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
StitcherImpl::start_geo_mappers (const SmartPtr<VKStitcher::StitcherParam> &param)
{
    uint32_t cam_num = _stitcher->get_camera_num ();

    for (uint32_t idx = 0; idx < cam_num; ++idx) {
        update_geomap_factors (idx);

        SmartPtr<GeoMapParam> &mapper_param = _res.mapper_param[idx];
        mapper_param->in_buf = param->in_bufs[idx];
        mapper_param->stitch_param = param;

        XCamReturn ret = _res.mapper[idx]->execute_buffer (mapper_param, false);
        XCAM_FAIL_RETURN (
            ERROR, xcam_ret_is_ok (ret), ret,
            "vk-stitcher(%s) execute geo mapper failed, idx:%d",
            XCAM_STR (_stitcher->get_name ()), idx);

#if DUMP_BUFFER
        dump_buf (mapper_param->out_buf, idx, "stitcher-geomap");
#endif
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
StitcherImpl::start_blender (
    const SmartPtr<VKStitcher::StitcherParam> &param, uint32_t idx)
{
    SmartPtr<VKBlender::Sync> &sync = _res.blender_sync[idx];
    if (!sync->is_synced ())
        return XCAM_RETURN_NO_ERROR;
    sync->reset ();

    SmartPtr<VKBlender::BlenderParam> &blend_param = _res.blender_param[idx];
    blend_param->out_buf = param->out_buf;

    XCamReturn ret = _res.blender[idx]->execute_buffer (blend_param, false);
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret,
        "vk-stitcher(%s) execute blender failed, idx:%d", XCAM_STR (_stitcher->get_name ()), idx);

#if DUMP_BUFFER
        dump_buf (param->out_buf, idx, "stitcher-blend");
#endif

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
StitcherImpl::start_blenders (const SmartPtr<VKStitcher::StitcherParam> &param, uint32_t idx)
{
    uint32_t cam_num = _stitcher->get_camera_num ();
    uint32_t pre_idx = (idx + cam_num - 1) % cam_num;

    XCamReturn ret = start_blender (param, pre_idx);
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret,
        "vk-stitcher(%s) start blender failed, idx:%d", XCAM_STR (_stitcher->get_name ()), pre_idx);

    ret = start_blender (param, idx);
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret,
        "vk-stitcher(%s) start blender failed, idx:%d", XCAM_STR (_stitcher->get_name ()), idx);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
StitcherImpl::start_copier (
    const SmartPtr<VKStitcher::StitcherParam> &param, uint32_t idx)
{
    uint32_t size = _stitcher->get_copy_area ().size ();

    for (uint32_t i = 0; i < size; ++i) {
        if(_res.copiers[i]->get_index () != idx)
            continue;

        SmartPtr<ImageHandler::Parameters> &copy_param = _res.copier_param[idx];
        copy_param->out_buf = param->out_buf;

        XCamReturn ret = _res.copiers[i]->execute_buffer (copy_param, false);
        XCAM_FAIL_RETURN (
            ERROR, xcam_ret_is_ok (ret), ret,
            "vk-stitcher(%s) execute copier failed, i:%d idx:%d",
            XCAM_STR (_stitcher->get_name ()), i, idx);

#if DUMP_BUFFER
        dump_buf (copy_param->out_buf, i, "stitcher-copy");
#endif
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
StitcherImpl::stop ()
{
    uint32_t cam_num = _stitcher->get_camera_num ();
    for (uint32_t i = 0; i < cam_num; ++i) {
        if (_res.mapper[i].ptr ()) {
            _res.mapper[i]->terminate ();
            _res.mapper[i].release ();
        }
        if (_res.mapper_pool[i].ptr ()) {
            _res.mapper_pool[i]->stop ();
        }

        if (_res.blender[i].ptr ()) {
            _res.blender[i]->terminate ();
            _res.blender[i].release ();
        }
    }

    for (Copiers::iterator i = _res.copiers.begin (); i != _res.copiers.end (); ++i) {
        SmartPtr<VKCopyHandler> &copier = *i;
        if (copier.ptr ()) {
            copier->terminate ();
            copier.release ();
        }
    }

    return XCAM_RETURN_NO_ERROR;
}

};

VKStitcher::VKStitcher (const SmartPtr<VKDevice> &dev, const char *name)
    : VKHandler (dev, name)
    , Stitcher (VK_STITCHER_ALIGNMENT_X, VK_STITCHER_ALIGNMENT_X)
{
    SmartPtr<VKSitcherPriv::StitcherImpl> impl = new VKSitcherPriv::StitcherImpl (this);
    XCAM_ASSERT (impl.ptr ());
    _impl = impl;
}

VKStitcher::~VKStitcher ()
{
    _impl.release ();
}

XCamReturn
VKStitcher::terminate ()
{
    _impl->stop ();
    return VKHandler::terminate ();
}

XCamReturn
VKStitcher::stitch_buffers (const VideoBufferList &in_bufs, SmartPtr<VideoBuffer> &out_buf)
{
    XCAM_FAIL_RETURN (
        ERROR, !in_bufs.empty (), XCAM_RETURN_ERROR_PARAM,
        "vk-stitcher(%s) input buffers is empty", XCAM_STR (get_name ()));

    SmartPtr<StitcherParam> param = new StitcherParam ();
    XCAM_ASSERT (param.ptr ());

    uint32_t buf_num = 0;
    for (VideoBufferList::const_iterator iter = in_bufs.begin(); iter != in_bufs.end (); ++iter) {
        XCAM_ASSERT ((*iter).ptr ());
        param->in_bufs[buf_num++] = *iter;
    }
    param->in_buf_num = buf_num;
    param->out_buf = out_buf;

    XCamReturn ret = execute_buffer (param, false);
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret,
        "vk-stitcher(%s) execute buffer failed", XCAM_STR (get_name ()));

    finish ();
    if (!out_buf.ptr ()) {
        out_buf = param->out_buf;
    }

    return ret;
}

XCamReturn
VKStitcher::configure_resource (const SmartPtr<Parameters> &param)
{
    XCAM_UNUSED (param);
    XCAM_ASSERT (_impl.ptr ());

    XCamReturn ret = estimate_round_slices ();
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret,
        "vk-stitcher(%s) estimate round view slices failed", XCAM_STR (get_name ()));

    ret = estimate_coarse_crops ();
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret,
        "vk-stitcher(%s) estimate coarse crops failed", XCAM_STR (get_name ()));

    ret = mark_centers ();
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret,
        "vk-stitcher(%s) mark centers failed", XCAM_STR (get_name ()));

    ret = estimate_overlap ();
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret,
        "vk-stitcher(%s) estimake coarse overlap failed", XCAM_STR (get_name ()));

    ret = update_copy_areas ();
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret,
        "vk-stitcher(%s) update copy areas failed", XCAM_STR (get_name ()));

    ret = _impl->init_resource ();
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret,
        "vk-stitcher(%s) initialize private config failed", XCAM_STR (get_name ()));

    VideoBufferInfo out_info;
    uint32_t out_width, out_height;
    get_output_size (out_width, out_height);
    XCAM_FAIL_RETURN (
        ERROR, out_width && out_height, XCAM_RETURN_ERROR_PARAM,
        "vk-stitcher(%s) output size was not set", XCAM_STR (get_name ()));

    out_info.init (
        V4L2_PIX_FMT_NV12, out_width, out_height,
        XCAM_ALIGN_UP (out_width, VK_STITCHER_ALIGNMENT_X),
        XCAM_ALIGN_UP (out_height, VK_STITCHER_ALIGNMENT_Y));
    set_out_video_info (out_info);

    return ret;
}

XCamReturn
VKStitcher::start_work (const SmartPtr<Parameters> &base)
{
    XCAM_ASSERT (base.ptr ());

    SmartPtr<StitcherParam> param = base.dynamic_cast_ptr<StitcherParam> ();
    XCAM_ASSERT (param.ptr () && param->in_buf_num > 0);

    XCamReturn ret = _impl->start_geo_mappers (param);
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), XCAM_RETURN_ERROR_PARAM,
        "vk_stitcher(%s) start geometry map failed", XCAM_STR (get_name ()));

    return ret;
}

void
VKStitcher::geomap_done (
    const SmartPtr<ImageHandler> &handler,
    const SmartPtr<ImageHandler::Parameters> &base, const XCamReturn error)
{
    XCAM_UNUSED (handler);
    XCAM_UNUSED (error);

    SmartPtr<VKSitcherPriv::GeoMapParam> param = base.dynamic_cast_ptr<VKSitcherPriv::GeoMapParam> ();
    XCAM_ASSERT (param.ptr ());
    SmartPtr<VKStitcher::StitcherParam> &stitch_param = param->stitch_param;
    XCAM_ASSERT (stitch_param.ptr ());

    _impl->update_blender_sync (param->idx);

    XCamReturn ret = _impl->start_blenders (stitch_param, param->idx);
    CHECK_RET (ret, "vk-stitcher(%s) start blenders failed, idx:%d", XCAM_STR (get_name ()), param->idx);

    ret = _impl->start_copier (stitch_param, param->idx);
    CHECK_RET (ret, "vk-stitcher(%s) start copier failed, idx:%d", XCAM_STR (get_name ()), param->idx);
}

SmartPtr<Stitcher>
Stitcher::create_vk_stitcher (const SmartPtr<VKDevice> dev)
{
    return new VKStitcher (dev);
}

}
