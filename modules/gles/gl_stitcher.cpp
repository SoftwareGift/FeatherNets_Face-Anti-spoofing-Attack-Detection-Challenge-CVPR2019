/*
 * gl_stitcher.cpp - GL stitcher implementation
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
 * Author: Yinhang Liu <yinhangx.liu@intel.com>
 */

#include "surview_fisheye_dewarp.h"
#include "gl_video_buffer.h"
#include "gl_geomap_handler.h"
#include "gl_blender.h"
#include "gl_copy_handler.h"
#include "gl_stitcher.h"

#define GL_STITCHER_ALIGNMENT_X 16
#define GL_STITCHER_ALIGNMENT_Y 4

#define MAP_FACTOR_X  16
#define MAP_FACTOR_Y  16

#define DUMP_BUFFER 0

namespace XCam {

#if DUMP_BUFFER
static void
dump_buf (const SmartPtr<VideoBuffer> buf, uint32_t idx, const char *prefix)
{
    XCAM_ASSERT (buf.ptr ());
    XCAM_ASSERT (prefix);

    char name[256];
    snprintf (name, 256, "%s-%d", prefix, idx);
    dump_buf_perfix_path (buf, name);
}
#else
static void
dump_buf (const SmartPtr<VideoBuffer> buf, ...) {
    XCAM_UNUSED (buf);
}
#endif

namespace GLSitcherPriv {

DECLARE_HANDLER_CALLBACK (CbGeoMap, GLStitcher, dewarp_done);
DECLARE_HANDLER_CALLBACK (CbBlender, GLStitcher, blender_done);
DECLARE_HANDLER_CALLBACK (CbCopier, GLStitcher, copier_done);

struct BlenderParam
    : GLBlender::BlenderParam
{
    SmartPtr<GLStitcher::StitcherParam>    stitch_param;
    uint32_t                               idx;

    BlenderParam (
        uint32_t i,
        const SmartPtr<VideoBuffer> &in0,
        const SmartPtr<VideoBuffer> &in1,
        const SmartPtr<VideoBuffer> &out)
        : GLBlender::BlenderParam (in0, in1, out)
        , idx (i)
    {}
};
typedef std::map<void*, SmartPtr<BlenderParam>> BlenderParams;

struct HandlerParam
    : ImageHandler::Parameters
{
    SmartPtr<GLStitcher::StitcherParam>    stitch_param;
    uint32_t                               idx;

    HandlerParam (uint32_t i)
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

struct Overlap {
    SmartPtr<GLBlender>    blender;
    BlenderParams          param_map;

    SmartPtr<BlenderParam> find_blender_param_in_map (
        const SmartPtr<GLStitcher::StitcherParam> &key,
        const uint32_t idx);
};

struct FisheyeDewarp {
    SmartPtr<GLGeoMapHandler>    dewarp;
    SmartPtr<BufferPool>         buf_pool;

    bool set_dewarp_factor ();
    XCamReturn set_dewarp_geo_table (
        const SmartPtr<GLGeoMapHandler> &mapper, const CameraInfo &cam_info,
        const Stitcher::RoundViewSlice &view_slice, const BowlDataConfig &bowl);
};

typedef std::vector<SmartPtr<GLCopyHandler>> Copiers;

class StitcherImpl {
    friend class XCam::GLStitcher;

public:
    StitcherImpl (GLStitcher *handler)
        : _stitcher (handler)
    {}

    XCamReturn init_config (uint32_t count);
    XCamReturn start_dewarps (const SmartPtr<GLStitcher::StitcherParam> &param);
    XCamReturn start_blenders (
        const SmartPtr<GLStitcher::StitcherParam> &param,
        uint32_t idx, const SmartPtr<VideoBuffer> &buf);
    XCamReturn start_copier (
        const SmartPtr<GLStitcher::StitcherParam> &param,
        uint32_t idx, const SmartPtr<VideoBuffer> &buf);

    XCamReturn start_single_blender (uint32_t idx, const SmartPtr<BlenderParam> &param);
    XCamReturn stop ();

    XCamReturn fisheye_dewarp_to_table ();

    const SmartPtr<GLComputeProgram> &get_sync_prog ();

private:
    SmartPtr<GLGeoMapHandler> create_geo_mapper (const Stitcher::RoundViewSlice &view_slice);

    XCamReturn init_fisheye (uint32_t idx);
    bool init_dewarp_factors (uint32_t idx);

private:
    FisheyeDewarp                 _fisheye[XCAM_STITCH_MAX_CAMERAS];
    Overlap                       _overlaps[XCAM_STITCH_MAX_CAMERAS];
    Copiers                       _copiers;

    Mutex                         _map_mutex;
    GLStitcher                   *_stitcher;
    SmartPtr<GLComputeProgram>    _sync_prog;
};

const SmartPtr<GLComputeProgram> &
StitcherImpl::get_sync_prog ()
{
    if (_sync_prog.ptr ())
        return _sync_prog;

    SmartPtr<GLComputeProgram> prog = GLComputeProgram::create_compute_program ("sync_program");
    XCAM_FAIL_RETURN (ERROR, prog.ptr (), NULL, "create sync program failed");
    _sync_prog = prog;

    return _sync_prog;
}

bool
StitcherImpl::init_dewarp_factors (uint32_t idx)
{
    XCAM_FAIL_RETURN (
        ERROR, _fisheye[idx].dewarp.ptr (), false,
        "FisheyeDewarp dewarp handler empty");

    Factor last_left_factor, last_right_factor, cur_left, cur_right;
    if (_stitcher->get_scale_mode () == ScaleSingleConst) {
        Factor unify_factor;
        _fisheye[idx].dewarp->get_factors (unify_factor.x, unify_factor.y);
        if (XCAM_DOUBLE_EQUAL_AROUND (unify_factor.x, 0.0f) ||
                XCAM_DOUBLE_EQUAL_AROUND (unify_factor.y, 0.0f)) { // not started.
            return true;
        }
        last_left_factor = last_right_factor = unify_factor;

        unify_factor.x = (last_left_factor.x + last_right_factor.x) / 2.0f;
        unify_factor.y = (last_left_factor.y + last_right_factor.y) / 2.0f;

        _fisheye[idx].dewarp->set_factors (unify_factor.x, unify_factor.y);
    } else {
        SmartPtr<GLDualConstGeoMapHandler> dewarp = _fisheye[idx].dewarp.dynamic_cast_ptr<GLDualConstGeoMapHandler> ();
        XCAM_ASSERT (dewarp.ptr ());

        dewarp->get_left_factors (last_left_factor.x, last_left_factor.y);
        dewarp->get_right_factors (last_right_factor.x, last_right_factor.y);
        if (XCAM_DOUBLE_EQUAL_AROUND (last_left_factor.x, 0.0f) ||
                XCAM_DOUBLE_EQUAL_AROUND (last_left_factor.y, 0.0f) ||
                XCAM_DOUBLE_EQUAL_AROUND (last_right_factor.y, 0.0f) ||
                XCAM_DOUBLE_EQUAL_AROUND (last_right_factor.y, 0.0f)) { // not started.
            return true;
        }

        dewarp->set_left_factors (last_left_factor.x, last_left_factor.y);
        dewarp->set_right_factors (last_right_factor.x, last_right_factor.y);
    }

    return true;
}


XCamReturn
FisheyeDewarp::set_dewarp_geo_table (
    const SmartPtr<GLGeoMapHandler> &mapper, const CameraInfo &cam_info,
    const Stitcher::RoundViewSlice &view_slice, const BowlDataConfig &bowl)
{
    PolyFisheyeDewarp fd;
    fd.set_intrinsic_param (cam_info.calibration.intrinsic);
    fd.set_extrinsic_param (cam_info.calibration.extrinsic);

    uint32_t table_width, table_height;
    table_width = view_slice.width / MAP_FACTOR_X;
    table_height = view_slice.height / MAP_FACTOR_Y;

    SurViewFisheyeDewarp::MapTable map_table(table_width * table_height);
    fd.fisheye_dewarp (
        map_table, table_width, table_height,
        view_slice.width, view_slice.height, bowl);

    XCAM_FAIL_RETURN (
        ERROR,
        mapper->set_lookup_table (map_table.data (), table_width, table_height),
        XCAM_RETURN_ERROR_UNKNOWN,
        "set fisheye dewarp lookup table failed");

    return XCAM_RETURN_NO_ERROR;
}

SmartPtr<GLGeoMapHandler>
StitcherImpl::create_geo_mapper (const Stitcher::RoundViewSlice &view_slice)
{
    XCAM_UNUSED (view_slice);

    SmartPtr<GLGeoMapHandler> dewarp;
    if (_stitcher->get_scale_mode () == ScaleSingleConst)
        dewarp = new GLGeoMapHandler ("sitcher_singleconst_remapper");
    else
        dewarp = new GLDualConstGeoMapHandler ("sitcher_dualconst_remapper");

    XCAM_ASSERT (dewarp.ptr ());
    return dewarp;
}

XCamReturn
StitcherImpl::init_fisheye (uint32_t idx)
{
    FisheyeDewarp &fisheye = _fisheye[idx];
    Stitcher::RoundViewSlice view_slice = _stitcher->get_round_view_slice (idx);

    SmartPtr<ImageHandler::Callback> dewarp_cb = new CbGeoMap (_stitcher);
    fisheye.dewarp = create_geo_mapper (view_slice);
    fisheye.dewarp->set_callback (dewarp_cb);

    VideoBufferInfo buf_info;
    buf_info.init (
        V4L2_PIX_FMT_NV12, view_slice.width, view_slice.height,
        XCAM_ALIGN_UP (view_slice.width, GL_STITCHER_ALIGNMENT_X),
        XCAM_ALIGN_UP (view_slice.height, GL_STITCHER_ALIGNMENT_Y));

    SmartPtr<BufferPool> pool = new GLVideoBufferPool (buf_info);
    XCAM_ASSERT (pool.ptr ());
    XCAM_FAIL_RETURN (
        ERROR, pool->reserve (XCAM_GL_RESERVED_BUF_COUNT), XCAM_RETURN_ERROR_MEM,
        "gl-stitcher(%s) reserve dewarp buffer pool failed, width:%d, height:%d",
        XCAM_STR (_stitcher->get_name ()), buf_info.width, buf_info.height);
    fisheye.buf_pool = pool;

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
StitcherImpl::init_config (uint32_t count)
{
    for (uint32_t i = 0; i < count; ++i) {
        XCamReturn ret = init_fisheye (i);
        XCAM_FAIL_RETURN (
            ERROR, xcam_ret_is_ok (ret), ret,
            "gl-stitcher(%s) init fisheye failed, idx:%d.", XCAM_STR (_stitcher->get_name ()), i);

        _overlaps[i].blender = create_gl_blender ().dynamic_cast_ptr<GLBlender>();
        XCAM_ASSERT (_overlaps[i].blender.ptr ());
        SmartPtr<ImageHandler::Callback> blender_cb = new CbBlender (_stitcher);
        XCAM_ASSERT (blender_cb.ptr ());
        _overlaps[i].blender->set_callback (blender_cb);
        _overlaps[i].param_map.clear ();

    }

    Stitcher::CopyAreaArray areas = _stitcher->get_copy_area ();
    uint32_t size = areas.size ();
    for (uint32_t i = 0; i < size; ++i) {
        XCAM_ASSERT (areas[i].in_idx < size);

        SmartPtr<ImageHandler::Callback> copier_cb = new CbCopier (_stitcher);
        XCAM_ASSERT (copier_cb.ptr ());
        SmartPtr<GLCopyHandler> copier = new GLCopyHandler ("stitch_copy");
        XCAM_ASSERT (copier.ptr ());

        copier->enable_allocator (false);
        copier->set_callback (copier_cb);
        copier->set_copy_area (areas[i].in_idx, areas[i].in_area, areas[i].out_area);
        _copiers.push_back (copier);
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
StitcherImpl::fisheye_dewarp_to_table ()
{
    uint32_t camera_num = _stitcher->get_camera_num ();
    for (uint32_t i = 0; i < camera_num; ++i) {
        CameraInfo cam_info;
        _stitcher->get_camera_info (i, cam_info);
        Stitcher::RoundViewSlice view_slice = _stitcher->get_round_view_slice (i);

        BowlDataConfig bowl = _stitcher->get_bowl_config ();
        bowl.angle_start = view_slice.hori_angle_start;
        bowl.angle_end = format_angle (view_slice.hori_angle_start + view_slice.hori_angle_range);

        uint32_t out_width, out_height;
        _stitcher->get_output_size (out_width, out_height);

        XCAM_ASSERT (_fisheye[i].dewarp.ptr ());
        _fisheye[i].dewarp->set_output_size (view_slice.width, view_slice.height);

        if (bowl.angle_end < bowl.angle_start)
            bowl.angle_start -= 360.0f;

        XCAM_LOG_INFO (
            "gl-stitcher(%s) camera(idx:%d) info(angle start:%.2f, range:%.2f), bowl info(angle start:%.2f, end:%.2f)",
            XCAM_STR (_stitcher->get_name ()), i,
            view_slice.hori_angle_start, view_slice.hori_angle_range,
            bowl.angle_start, bowl.angle_end);

        XCamReturn ret = _fisheye[i].set_dewarp_geo_table (_fisheye[i].dewarp, cam_info, view_slice, bowl);

        XCAM_FAIL_RETURN (
            ERROR, xcam_ret_is_ok (ret), ret,
            "gl-stitcher(%s) set dewarp geo table failed, idx:%d", XCAM_STR (_stitcher->get_name ()), i);
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
StitcherImpl::start_dewarps (const SmartPtr<GLStitcher::StitcherParam> &param)
{
    uint32_t camera_num = _stitcher->get_camera_num ();

    for (uint32_t i = 0; i < camera_num; ++i) {
        SmartPtr<VideoBuffer> out_buf = _fisheye[i].buf_pool->get_buffer ();
        SmartPtr<HandlerParam> dewarp_params = new HandlerParam (i);
        dewarp_params->in_buf = param->in_bufs[i];
        dewarp_params->out_buf = out_buf;
        dewarp_params->stitch_param = param;

        init_dewarp_factors (i);
        XCamReturn ret = _fisheye[i].dewarp->execute_buffer (dewarp_params, false);
        XCAM_FAIL_RETURN (
            ERROR, xcam_ret_is_ok (ret), ret,
            "gl-stitcher(%s) fisheye dewarp buffer failed, idx:%d",
            XCAM_STR (_stitcher->get_name ()), i);
    }

    return XCAM_RETURN_NO_ERROR;
}

SmartPtr<BlenderParam>
Overlap::find_blender_param_in_map (
    const SmartPtr<GLStitcher::StitcherParam> &key, uint32_t idx)
{
    SmartPtr<BlenderParam> param;
    BlenderParams::iterator i = param_map.find (key.ptr ());
    if (i == param_map.end ()) {
        param = new BlenderParam (idx, NULL, NULL, NULL);
        XCAM_ASSERT (param.ptr ());
        param->stitch_param = key;
        param_map.insert (std::make_pair ((void*)key.ptr (), param));
    } else {
        param = (*i).second;
    }

    return param;
}

XCamReturn
StitcherImpl::start_single_blender (
    uint32_t idx, const SmartPtr<BlenderParam> &param)
{
    SmartPtr<GLBlender> blender = _overlaps[idx].blender;
    const Stitcher::ImageOverlapInfo &overlap_info = _stitcher->get_overlap (idx);
    uint32_t out_width, out_height;
    _stitcher->get_output_size (out_width, out_height);

    blender->set_output_size (out_width, out_height);
    blender->set_merge_window (overlap_info.out_area);
    blender->set_input_valid_area (overlap_info.left, 0);
    blender->set_input_valid_area (overlap_info.right, 1);
    blender->set_input_merge_area (overlap_info.left, 0);
    blender->set_input_merge_area (overlap_info.right, 1);

    return blender->execute_buffer (param, false);
}

XCamReturn
StitcherImpl::start_blenders (
    const SmartPtr<GLStitcher::StitcherParam> &param,
    uint32_t idx, const SmartPtr<VideoBuffer> &buf)
{
    SmartPtr<BlenderParam> cur_param, prev_param;
    const uint32_t camera_num = _stitcher->get_camera_num ();
    uint32_t pre_idx = (idx + camera_num - 1) % camera_num;

    {
        SmartPtr<BlenderParam> param_b;

        SmartLock locker (_map_mutex);
        param_b = _overlaps[idx].find_blender_param_in_map (param, idx);
        param_b->in_buf = buf;
        if (param_b->in_buf.ptr () && param_b->in1_buf.ptr ()) {
            cur_param = param_b;
            _overlaps[idx].param_map.erase (param.ptr ());
        }

        param_b = _overlaps[pre_idx].find_blender_param_in_map (param, pre_idx);
        param_b->in1_buf = buf;
        if (param_b->in_buf.ptr () && param_b->in1_buf.ptr ()) {
            prev_param = param_b;
            _overlaps[pre_idx].param_map.erase (param.ptr ());
        }
    }

    if (cur_param.ptr ()) {
        cur_param->out_buf = param->out_buf;
        XCamReturn ret = start_single_blender (idx, cur_param);
        XCAM_FAIL_RETURN (
            ERROR, xcam_ret_is_ok (ret), ret,
            "gl-stitcher(%s) blend overlap idx:%d failed", XCAM_STR (_stitcher->get_name ()), idx);
    }

    if (prev_param.ptr ()) {
        prev_param->out_buf = param->out_buf;
        XCamReturn ret = start_single_blender (pre_idx, prev_param);
        XCAM_FAIL_RETURN (
            ERROR, xcam_ret_is_ok (ret), ret,
            "gl-stitcher(%s) blend overlap idx:%d failed", XCAM_STR (_stitcher->get_name ()), pre_idx);
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
StitcherImpl::start_copier (
    const SmartPtr<GLStitcher::StitcherParam> &param,
    uint32_t idx, const SmartPtr<VideoBuffer> &buf)
{
    XCAM_ASSERT (param.ptr ());
    XCAM_ASSERT (buf.ptr ());

    uint32_t size = _stitcher->get_copy_area ().size ();
    XCAM_FAIL_RETURN (
        ERROR, idx <= size, XCAM_RETURN_ERROR_PARAM,
        "gl-stitcher(%s) invalid idx:%d", XCAM_STR (_stitcher->get_name ()), idx);

    for (uint32_t i = 0; i < size; ++i) {
        if(_copiers[i]->get_index () != idx)
            continue;

        SmartPtr<HandlerParam> copy_params = new HandlerParam (i);
        copy_params->in_buf = buf;
        copy_params->out_buf = param->out_buf;
        copy_params->stitch_param = param;

        XCamReturn ret = _copiers[i]->execute_buffer (copy_params, false);
        XCAM_FAIL_RETURN (
            ERROR, xcam_ret_is_ok (ret), ret,
            "gl-stitcher(%s) execute copier failed, i:%d idx:%d",
            XCAM_STR (_stitcher->get_name ()), i, idx);
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
StitcherImpl::stop ()
{
    uint32_t cam_num = _stitcher->get_camera_num ();
    for (uint32_t i = 0; i < cam_num; ++i) {
        if (_fisheye[i].dewarp.ptr ()) {
            _fisheye[i].dewarp->terminate ();
            _fisheye[i].dewarp.release ();
        }
        if (_fisheye[i].buf_pool.ptr ()) {
            _fisheye[i].buf_pool->stop ();
        }

        if (_overlaps[i].blender.ptr ()) {
            _overlaps[i].blender->terminate ();
            _overlaps[i].blender.release ();
        }
    }

    for (Copiers::iterator i_copier = _copiers.begin (); i_copier != _copiers.end (); ++i_copier) {
        SmartPtr<GLCopyHandler> &copier = *i_copier;
        if (copier.ptr ()) {
            copier->terminate ();
            copier.release ();
        }
    }

    return XCAM_RETURN_NO_ERROR;
}

};

GLStitcher::GLStitcher (const char *name)
    : GLImageHandler (name)
    , Stitcher (GL_STITCHER_ALIGNMENT_X, GL_STITCHER_ALIGNMENT_X)
{
    SmartPtr<GLSitcherPriv::StitcherImpl> impl = new GLSitcherPriv::StitcherImpl (this);
    XCAM_ASSERT (impl.ptr ());
    _impl = impl;
}

GLStitcher::~GLStitcher ()
{
}

XCamReturn
GLStitcher::terminate ()
{
    _impl->stop ();
    return GLImageHandler::terminate ();
}

XCamReturn
GLStitcher::stitch_buffers (const VideoBufferList &in_bufs, SmartPtr<VideoBuffer> &out_buf)
{
    XCAM_FAIL_RETURN (
        ERROR, !in_bufs.empty (), XCAM_RETURN_ERROR_PARAM,
        "gl-stitcher(%s) stitch buffer failed, input buffers is empty", XCAM_STR (get_name ()));

    SmartPtr<StitcherParam> param = new StitcherParam;
    XCAM_ASSERT (param.ptr ());
    param->out_buf = out_buf;

    uint32_t count = 0;
    for (VideoBufferList::const_iterator iter = in_bufs.begin(); iter != in_bufs.end (); ++iter) {
        SmartPtr<VideoBuffer> buf = *iter;
        XCAM_ASSERT (buf.ptr ());
        param->in_bufs[count++] = buf;
    }
    param->in_buf_num = count;

    XCamReturn ret = execute_buffer (param, false);
    if (!out_buf.ptr () && xcam_ret_is_ok (ret)) {
        out_buf = param->out_buf;
    }

    return ret;
}

XCamReturn
GLStitcher::configure_resource (const SmartPtr<Parameters> &param)
{
    XCAM_UNUSED (param);
    XCAM_ASSERT (_impl.ptr ());

    XCamReturn ret = estimate_round_slices ();
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret,
        "gl-stitcher(%s) estimate round view slices failed", XCAM_STR (get_name ()));

    ret = estimate_coarse_crops ();
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret,
        "gl-stitcher(%s) estimate coarse crops failed", XCAM_STR (get_name ()));

    ret = mark_centers ();
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret,
        "gl-stitcher(%s) mark centers failed", XCAM_STR (get_name ()));

    ret = estimate_overlap ();
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret,
        "gl-stitcher(%s) estimake coarse overlap failed", XCAM_STR (get_name ()));

    ret = update_copy_areas ();
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret,
        "gl-stitcher(%s) update copy areas failed", XCAM_STR (get_name ()));

    uint32_t camera_count = get_camera_num ();
    ret = _impl->init_config (camera_count);
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret,
        "gl-stitcher(%s) initialize private config failed", XCAM_STR (get_name ()));

    ret = _impl->fisheye_dewarp_to_table ();
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret,
        "gl-stitcher(%s) fisheye_dewarp_to_table failed", XCAM_STR (get_name ()));

    VideoBufferInfo out_info;
    uint32_t out_width, out_height;
    get_output_size (out_width, out_height);
    XCAM_FAIL_RETURN (
        ERROR, out_width && out_height, XCAM_RETURN_ERROR_PARAM,
        "gl-stitcher(%s) output size was not set", XCAM_STR (get_name ()));

    out_info.init (
        V4L2_PIX_FMT_NV12, out_width, out_height,
        XCAM_ALIGN_UP (out_width, GL_STITCHER_ALIGNMENT_X),
        XCAM_ALIGN_UP (out_height, GL_STITCHER_ALIGNMENT_Y));
    set_out_video_info (out_info);

    return ret;
}

XCamReturn
GLStitcher::start_work (const SmartPtr<Parameters> &base)
{
    XCAM_ASSERT (base.ptr ());

    SmartPtr<StitcherParam> param = base.dynamic_cast_ptr<StitcherParam> ();
    XCAM_FAIL_RETURN (
        ERROR, param.ptr () && param->in_buf_num > 0 && param->in_bufs[0].ptr (), XCAM_RETURN_ERROR_PARAM,
        "gl-stitcher(%s) start work failed, invalid parameters", XCAM_STR (get_name ()));

    XCamReturn ret = _impl->start_dewarps (param);
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), XCAM_RETURN_ERROR_PARAM,
        "gl_stitcher(%s) start dewarps failed", XCAM_STR (get_name ()));

    const SmartPtr<GLComputeProgram> prog = _impl->get_sync_prog ();
    XCAM_ASSERT (prog.ptr ());
    ret = prog->finish ();

    return ret;
}

void
GLStitcher::dewarp_done (
    const SmartPtr<ImageHandler> &handler,
    const SmartPtr<ImageHandler::Parameters> &base, const XCamReturn error)
{
    XCAM_UNUSED (handler);

    SmartPtr<GLSitcherPriv::HandlerParam> dewarp_param = base.dynamic_cast_ptr<GLSitcherPriv::HandlerParam> ();
    XCAM_ASSERT (dewarp_param.ptr ());
    SmartPtr<GLStitcher::StitcherParam> param = dewarp_param->stitch_param;
    XCAM_ASSERT (param.ptr ());

    execute_done (param, error);

    XCAM_LOG_INFO ("gl-stitcher(%s) camera(idx:%d) dewarp done", XCAM_STR (get_name ()), dewarp_param->idx);
    dump_buf (dewarp_param->out_buf, dewarp_param->idx, "stitcher-dewarp");

    XCamReturn ret = _impl->start_blenders (param, dewarp_param->idx, dewarp_param->out_buf);
    if (!xcam_ret_is_ok (ret))
        XCAM_LOG_ERROR ("start_blenders failed");

    ret = _impl->start_copier (param, dewarp_param->idx, dewarp_param->out_buf);
    if (!xcam_ret_is_ok (ret))
        XCAM_LOG_ERROR ("start_copier failed");
}

void
GLStitcher::blender_done (
    const SmartPtr<ImageHandler> &handler,
    const SmartPtr<ImageHandler::Parameters> &base, const XCamReturn error)
{
    XCAM_UNUSED (handler);

    SmartPtr<GLSitcherPriv::BlenderParam> blender_param = base.dynamic_cast_ptr<GLSitcherPriv::BlenderParam> ();
    XCAM_ASSERT (blender_param.ptr ());
    SmartPtr<GLStitcher::StitcherParam> param = blender_param->stitch_param;
    XCAM_ASSERT (param.ptr ());

    execute_done (param, error);

    XCAM_LOG_INFO ("gl-stitcher(%s) overlap:%d done", XCAM_STR (handler->get_name ()), blender_param->idx);
    dump_buf (blender_param->out_buf, blender_param->idx, "stitcher-blend");
}

void
GLStitcher::copier_done (
    const SmartPtr<ImageHandler> &handler,
    const SmartPtr<ImageHandler::Parameters> &base, const XCamReturn error)
{
    XCAM_UNUSED (handler);

    SmartPtr<GLSitcherPriv::HandlerParam> copy_param = base.dynamic_cast_ptr<GLSitcherPriv::HandlerParam> ();
    XCAM_ASSERT (copy_param.ptr ());
    SmartPtr<GLStitcher::StitcherParam> param = copy_param->stitch_param;
    XCAM_ASSERT (param.ptr ());

    execute_done (param, error);

    XCAM_LOG_INFO ("gl-stitcher(%s) camera(idx:%d) copy done", XCAM_STR (get_name ()), copy_param->idx);
}

SmartPtr<Stitcher>
Stitcher::create_gl_stitcher ()
{
    return new GLStitcher;
}

}
