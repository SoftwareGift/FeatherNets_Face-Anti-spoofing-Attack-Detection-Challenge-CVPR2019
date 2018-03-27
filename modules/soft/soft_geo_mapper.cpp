/*
 * soft_geo_mapper.cpp - soft geometry mapper implementation
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

#include "soft_geo_mapper.h"
#include "soft_geo_tasks_priv.h"

#define XCAM_GEO_MAP_ALIGNMENT_X 8
#define XCAM_GEO_MAP_ALIGNMENT_Y 2

namespace XCam {

DECLARE_WORK_CALLBACK (CbGeoMapTask, SoftGeoMapper, remap_task_done);
DECLARE_WORK_CALLBACK (CbGeoMapDualConstTask, SoftDualConstGeoMapper, remap_task_done);

SoftGeoMapper::SoftGeoMapper (const char *name)
    : SoftHandler (name)
{
}

SoftGeoMapper::~SoftGeoMapper ()
{
}

bool
SoftGeoMapper::set_lookup_table (const PointFloat2 *data, uint32_t width, uint32_t height)
{
    XCAM_FAIL_RETURN(
        ERROR, width > 1 && height > 1 && data, false,
        "SoftGeoMapper(%s) set loop up table need w>1 and h>1, but width:%d, height:%d",
        XCAM_STR (get_name ()), width, height);

    _lookup_table = new Float2Image (width, height);

    XCAM_FAIL_RETURN(
        ERROR, _lookup_table.ptr () && _lookup_table->is_valid (), false,
        "SoftGeoMapper(%s) set loop up table failed in data allocation",
        XCAM_STR (get_name ()));

    for (uint32_t i = 0; i < height; ++i) {
        Float2 *ret = _lookup_table->get_buf_ptr (0, i);
        const PointFloat2 *line = &data[i * width];
        for (uint32_t j = 0; j < width; ++j) {
            ret[j].x = line [j].x;
            ret[j].y = line [j].y;
        }
    }

    return true;
}

XCamReturn
SoftGeoMapper::remap (
    const SmartPtr<VideoBuffer> &in,
    SmartPtr<VideoBuffer> &out_buf)
{
    SmartPtr<ImageHandler::Parameters> param = new ImageHandler::Parameters (in, out_buf);
    XCamReturn ret = execute_buffer (param, true);
    if (xcam_ret_is_ok (ret) && !out_buf.ptr ()) {
        out_buf = param->out_buf;
    }

    return ret;
}

XCamReturn
SoftGeoMapper::configure_resource (const SmartPtr<Parameters> &param)
{
    XCAM_FAIL_RETURN(
        ERROR, _lookup_table.ptr () && _lookup_table->is_valid (), XCAM_RETURN_ERROR_PARAM,
        "SoftGeoMapper(%s) configure failed, look_up_table was not set correctly",
        XCAM_STR (get_name ()));

    const VideoBufferInfo &in_info = param->in_buf->get_video_info ();
    XCAM_FAIL_RETURN (
        ERROR, in_info.format == V4L2_PIX_FMT_NV12, XCAM_RETURN_ERROR_PARAM,
        "SoftGeoMapper(%s) only support format(NV12) but input format is %s",
        XCAM_STR(get_name ()), xcam_fourcc_to_string (in_info.format));

    uint32_t width, height;
    get_output_size (width, height);
    VideoBufferInfo out_info;
    out_info.init (
        in_info.format, width, height,
        XCAM_ALIGN_UP (width, XCAM_GEO_MAP_ALIGNMENT_X),
        XCAM_ALIGN_UP (height, XCAM_GEO_MAP_ALIGNMENT_Y));
    set_out_video_info (out_info);

    init_factors ();

    XCAM_ASSERT (!_map_task.ptr ());
    _map_task = create_remap_task ();

    return XCAM_RETURN_NO_ERROR;
}

void
SoftGeoMapper::set_work_size (
    uint32_t thread_x, uint32_t thread_y,
    uint32_t luma_width, uint32_t luma_height)
{
    WorkSize work_unit = _map_task->get_work_uint ();
    WorkSize global_size (
        xcam_ceil (luma_width, work_unit.value[0]) / work_unit.value[0],
        xcam_ceil (luma_height, work_unit.value[1]) / work_unit.value[1]);
    WorkSize local_size (
        xcam_ceil(global_size.value[0], thread_x) / thread_x ,
        xcam_ceil(global_size.value[1], thread_y) / thread_y);

    _map_task->set_local_size (local_size);
    _map_task->set_global_size (global_size);
}

bool
SoftGeoMapper::init_factors ()
{
    Float2 factors;
    get_factors (factors.x, factors.y);
    if (!XCAM_DOUBLE_EQUAL_AROUND (factors.x, 0.0f) && !XCAM_DOUBLE_EQUAL_AROUND (factors.y, 0.0f))
        return true;

    return auto_calculate_factors (_lookup_table->get_width (), _lookup_table->get_height ());
}

SmartPtr<XCamSoftTasks::GeoMapTask>
SoftGeoMapper::create_remap_task ()
{
    SmartPtr<XCamSoftTasks::GeoMapTask> map_task = new XCamSoftTasks::GeoMapTask (new CbGeoMapTask (this));
    XCAM_ASSERT (map_task.ptr ());

    return map_task;
}

XCamReturn
SoftGeoMapper::start_remap_task (const SmartPtr<ImageHandler::Parameters> &param)
{
    XCAM_ASSERT (_map_task.ptr ());
    XCAM_ASSERT (_lookup_table.ptr ());

    Float2 factors;
    get_factors (factors.x, factors.y);

    SmartPtr<VideoBuffer> in_buf = param->in_buf, out_buf = param->out_buf;
    SmartPtr<XCamSoftTasks::GeoMapTask::Args> args = new XCamSoftTasks::GeoMapTask::Args (param);
    args->in_luma = new UcharImage (in_buf, 0);
    args->in_uv = new Uchar2Image (in_buf, 1);
    args->out_luma = new UcharImage (out_buf, 0);
    args->out_uv = new Uchar2Image (out_buf, 1);
    args->lookup_table = _lookup_table;
    args->factors = factors;

    set_work_size (2, 2, args->out_luma->get_width (), args->out_luma->get_height ());

    param->in_buf.release ();
    return _map_task->work (args);
}

XCamReturn
SoftGeoMapper::start_work (const SmartPtr<ImageHandler::Parameters> &param)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    XCAM_ASSERT (param->out_buf.ptr ());

    ret = start_remap_task (param);
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret,
        "SoftGeoMapper(%s) start_work failed on idx0", XCAM_STR (get_name ()));

    param->in_buf.release ();

    return ret;
};

XCamReturn
SoftGeoMapper::terminate ()
{
    if (_map_task.ptr ()) {
        _map_task->stop ();
        _map_task.release ();
    }
    return SoftHandler::terminate ();
}

void
SoftGeoMapper::remap_task_done (
    const SmartPtr<Worker> &worker, const SmartPtr<Worker::Arguments> &base, const XCamReturn error)
{
    XCAM_UNUSED (worker);
    XCAM_ASSERT (worker.ptr () == _map_task.ptr ());

    SmartPtr<XCamSoftTasks::GeoMapTask::Args> args = base.dynamic_cast_ptr<XCamSoftTasks::GeoMapTask::Args> ();
    XCAM_ASSERT (args.ptr ());

    const SmartPtr<ImageHandler::Parameters> param = args->get_param ();
    if (!check_work_continue (param, error))
        return;

    work_well_done (param, error);
}

SmartPtr<SoftHandler> create_soft_geo_mapper ()
{
    SmartPtr<SoftHandler> mapper = new SoftGeoMapper ();
    XCAM_ASSERT (mapper.ptr ());

    return mapper;
}

SmartPtr<GeoMapper>
GeoMapper::create_soft_geo_mapper ()
{
    SmartPtr<SoftHandler> handler = XCam::create_soft_geo_mapper ();
    return handler.dynamic_cast_ptr<GeoMapper> ();
}

SoftDualConstGeoMapper::SoftDualConstGeoMapper (const char *name)
    : SoftGeoMapper (name)
    , _left_factor_x (0.0f)
    , _left_factor_y (0.0f)
    , _right_factor_x (0.0f)
    , _right_factor_y (0.0f)
{
}

SoftDualConstGeoMapper::~SoftDualConstGeoMapper ()
{
}

bool
SoftDualConstGeoMapper::set_left_factors (float x, float y)
{
    XCAM_FAIL_RETURN (
        ERROR, !XCAM_DOUBLE_EQUAL_AROUND (x, 0.0f) && !XCAM_DOUBLE_EQUAL_AROUND (y, 0.0f), false,
        "SoftGeoMapper(%s) set factors failed. (x:%.3f, h:%.3f)", XCAM_STR(get_name ()), x, y);
    _left_factor_x = x;
    _left_factor_y = y;

    return true;
}

bool
SoftDualConstGeoMapper::set_right_factors (float x, float y)
{
    XCAM_FAIL_RETURN (
        ERROR, !XCAM_DOUBLE_EQUAL_AROUND (x, 0.0f) && !XCAM_DOUBLE_EQUAL_AROUND (y, 0.0f), false,
        "SoftGeoMapper(%s) set factors failed. (x:%.3f, h:%.3f)", XCAM_STR(get_name ()), x, y);
    _right_factor_x = x;
    _right_factor_y = y;

    return true;
}

bool
SoftDualConstGeoMapper::auto_calculate_factors (uint32_t lut_w, uint32_t lut_h)
{
    XCAM_FAIL_RETURN (
        ERROR, lut_w > 1 && lut_w > 1, false,
        "SoftGeoMapper(%s) auto calculate factors failed. lookuptable size need > 1. but set with (w:%d, h:%d)",
        XCAM_STR(get_name ()), lut_w, lut_h);

    uint32_t width, height;
    get_output_size (width, height);
    XCAM_FAIL_RETURN (
        ERROR, width > 1 && height > 1, false,
        "SoftGeoMapper(%s) auto calculate factors failed. output size was not set. (w:%d, h:%d)",
        XCAM_STR(get_name ()), width, height);

    _left_factor_x = (width - 1.0f) / (lut_w - 1.0f);
    _left_factor_y = (height - 1.0f) / (lut_h - 1.0f);

    _right_factor_x = _left_factor_x;
    _right_factor_y = _left_factor_y;

    return true;
}

bool
SoftDualConstGeoMapper::init_factors ()
{
    Float2 left_factors, right_factors;
    get_left_factors (left_factors.x, left_factors.y);
    get_right_factors (right_factors.x, right_factors.y);
    if (!XCAM_DOUBLE_EQUAL_AROUND (left_factors.x, 0.0f) && !XCAM_DOUBLE_EQUAL_AROUND (left_factors.y, 0.0f) &&
            !XCAM_DOUBLE_EQUAL_AROUND (right_factors.x, 0.0f) && !XCAM_DOUBLE_EQUAL_AROUND (right_factors.y, 0.0f))
        return true;

    SmartPtr<Float2Image> lookup_table = get_lookup_table ();
    XCAM_ASSERT (lookup_table.ptr ());

    return auto_calculate_factors (lookup_table->get_width (), lookup_table->get_height ());
}

SmartPtr<XCamSoftTasks::GeoMapTask>
SoftDualConstGeoMapper::create_remap_task ()
{
    SmartPtr<XCamSoftTasks::GeoMapTask> map_task =
        new XCamSoftTasks::GeoMapDualConstTask (new CbGeoMapDualConstTask (this));
    XCAM_ASSERT (map_task.ptr ());

    return map_task;
}

XCamReturn
SoftDualConstGeoMapper::prepare_arguments (
    const SmartPtr<Worker::Arguments> &base,
    const SmartPtr<ImageHandler::Parameters> &param)
{
    SmartPtr<Float2Image> lookup_table = get_lookup_table ();
    XCAM_ASSERT (lookup_table.ptr ());

    SmartPtr<VideoBuffer> in_buf = param->in_buf, out_buf = param->out_buf;
    SmartPtr<XCamSoftTasks::GeoMapDualConstTask::Args> args =
        base.dynamic_cast_ptr<XCamSoftTasks::GeoMapDualConstTask::Args> ();
    XCAM_ASSERT (args.ptr ());

    Float2 factors;
    get_left_factors (factors.x, factors.y);
    args->left_factor = factors;
    get_right_factors (factors.x, factors.y);
    args->right_factor = factors;
    args->in_luma = new UcharImage (in_buf, 0);
    args->in_uv = new Uchar2Image (in_buf, 1);
    args->out_luma = new UcharImage (out_buf, 0);
    args->out_uv = new Uchar2Image (out_buf, 1);
    args->lookup_table = lookup_table;

    set_work_size (2, 2, args->out_luma->get_width (), args->out_luma->get_height ());

    param->in_buf.release ();
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
SoftDualConstGeoMapper::start_remap_task (const SmartPtr<ImageHandler::Parameters> &param)
{
    SmartPtr<XCamSoftTasks::GeoMapTask> map_task = get_map_task ();
    XCAM_ASSERT (map_task.ptr ());

    SmartPtr<XCamSoftTasks::GeoMapDualConstTask::Args> args =
        new XCamSoftTasks::GeoMapDualConstTask::Args (param);
    XCAM_ASSERT (args.ptr ());

    prepare_arguments (args, param);

    return map_task->work (args);
}

void
SoftDualConstGeoMapper::remap_task_done (
    const SmartPtr<Worker> &worker, const SmartPtr<Worker::Arguments> &base, const XCamReturn error)
{
    XCAM_UNUSED (worker);
    XCAM_ASSERT (worker.ptr () == get_map_task().ptr ());

    SmartPtr<XCamSoftTasks::GeoMapDualConstTask::Args> args =
        base.dynamic_cast_ptr<XCamSoftTasks::GeoMapDualConstTask::Args> ();
    XCAM_ASSERT (args.ptr ());

    const SmartPtr<ImageHandler::Parameters> param = args->get_param ();
    if (!check_work_continue (param, error))
        return;

    work_well_done (param, error);
}

}
