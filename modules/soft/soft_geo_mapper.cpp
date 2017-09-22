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
#include "soft_worker.h"

#define XCAM_GEO_MAP_ALIGNMENT_X 8
#define XCAM_GEO_MAP_ALIGNMENT_Y 2

namespace XCam {

DECLARE_WORK_CALLBACK (CbGeoMapTask, SoftGeoMapper, remap_task_done);

namespace XCamSoftTasks {

class GeoMapTask
    : public SoftWorker
{
public:
    struct Args : SoftArgs {
        SmartPtr<UcharImage>        in_luma, out_luma;
        SmartPtr<Uchar2Image>       in_uv, out_uv;
        SmartPtr<Float2Image>       lookup_table;
        Float2                      factors;

        Args (
            const SmartPtr<ImageHandler::Parameters> &param)
            : SoftArgs (param)
        {}
    };

public:
    explicit GeoMapTask (const SmartPtr<Worker::Callback> &cb)
        : SoftWorker ("GeoMapTask", cb)
    {
        set_work_uint (8, 2);
    }

private:
    virtual XCamReturn work_range (const SmartPtr<Arguments> &args, const WorkRange &range);
};

}

SoftGeoMapper::SoftGeoMapper (const char *name)
    : SoftHandler (name)
{
}

SoftGeoMapper::~SoftGeoMapper ()
{
}

bool
SoftGeoMapper::set_lookup_table (GeoData *data, uint32_t width, uint32_t height)
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
        GeoData *line = &data[i * width];
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
        "SoftGeoMapper(:%s) only support format(NV12) but input format is %s",
        XCAM_STR(get_name ()), xcam_fourcc_to_string (in_info.format));

    Float2 factors;
    get_factors (factors.x, factors.y);
    if (XCAM_DOUBLE_EQUAL_AROUND (factors.x, 0.0f) ||
            XCAM_DOUBLE_EQUAL_AROUND (factors.y, 0.0f)) {
        auto_calculate_factors (_lookup_table->get_width (), _lookup_table->get_height ());
    }

    uint32_t width, height;
    get_output_size (width, height);
    VideoBufferInfo out_info;
    out_info.init (
        in_info.format, width, height,
        XCAM_ALIGN_UP (width, XCAM_GEO_MAP_ALIGNMENT_X),
        XCAM_ALIGN_UP (height, XCAM_GEO_MAP_ALIGNMENT_Y));
    set_out_video_info (out_info);

    XCAM_ASSERT (!_map_task.ptr ());
    _map_task = new XCamSoftTasks::GeoMapTask (new CbGeoMapTask(this));
    XCAM_ASSERT (_map_task.ptr ());

    return XCAM_RETURN_NO_ERROR;
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

    uint32_t thread_x = 2, thread_y = 2;
    WorkSize work_unit = _map_task->get_work_uint ();
    WorkSize global_size (
        xcam_ceil (args->out_luma->get_width (), work_unit.value[0]) / work_unit.value[0],
        xcam_ceil (args->out_luma->get_height (), work_unit.value[1]) / work_unit.value[1]);
    WorkSize local_size (
        xcam_ceil(global_size.value[0], thread_x) / thread_x ,
        xcam_ceil(global_size.value[1], thread_y) / thread_y);

    _map_task->set_local_size (local_size);
    _map_task->set_global_size (global_size);

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
        "geo_mapper:%s start_work failed on idx0", XCAM_STR (get_name ()));

    param->in_buf.release ();

    return ret;
};

void
SoftGeoMapper::remap_task_done (
    const SmartPtr<Worker> &worker, const SmartPtr<Worker::Arguments> &base, const XCamReturn error)
{
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

XCamReturn
XCamSoftTasks::GeoMapTask::work_range (const SmartPtr<Arguments> &base, const WorkRange &range)
{

    SmartPtr<GeoMapTask::Args> args = base.dynamic_cast_ptr<GeoMapTask::Args> ();
    XCAM_ASSERT (args.ptr ());
    UcharImage *in_luma = args->in_luma.ptr (), *out_luma = args->out_luma.ptr ();
    Uchar2Image *in_uv = args->in_uv.ptr (), *out_uv = args->out_uv.ptr ();
    Float2Image *lut = args->lookup_table.ptr ();
    XCAM_ASSERT (in_luma && in_uv);
    XCAM_ASSERT (out_luma && out_uv);
    XCAM_ASSERT (lut);

    Float2 factors = args->factors;
    XCAM_ASSERT (
        !XCAM_DOUBLE_EQUAL_AROUND (factors.x, 0.0f) &&
        !XCAM_DOUBLE_EQUAL_AROUND (factors.y, 0.0f));

    Float2 out_center ((out_luma->get_width () - 1.0f ) / 2.0f, (out_luma->get_height () - 1.0f ) / 2.0f);
    Float2 lut_center ((lut->get_width () - 1.0f) / 2.0f, (lut->get_height () - 1.0f) / 2.0f);
    float x_step = 1.0f / factors.x;
    float y_step = 1.0f / factors.y;

    for (uint32_t y = range.pos[1]; y < range.pos[1] + range.pos_len[1]; ++y)
        for (uint32_t x = range.pos[0]; x < range.pos[0] + range.pos_len[0]; ++x)
        {
            // calculate 8x2 luma, center aligned
            Float2 in_pos[8];
            float  luma_value[8];
            Uchar  luma_uc[8];
            uint32_t out_x = x * 8, out_y = y * 2;

            //1st-line luma
            Float2 out_pos (out_x, out_y);
            out_pos -= out_center;
            Float2 first = out_pos / factors;
            first += lut_center;
            Float2 lut_pos[8] = {
                first, Float2(first.x + x_step, first.y),
                Float2(first.x + x_step * 2, first.y), Float2(first.x + x_step * 3, first.y),
                Float2(first.x + x_step * 4, first.y), Float2(first.x + x_step * 5, first.y),
                Float2(first.x + x_step * 6, first.y), Float2(first.x + x_step * 7, first.y)
            };
            lut->read_interpolate_array<Float2, 8> (lut_pos, in_pos);
            in_luma->read_interpolate_array<float, 8> (in_pos, luma_value);
            convert_to_uchar_N<float, 8> (luma_value, luma_uc);
            out_luma->write_array_no_check<8> (out_x, out_y, luma_uc);

            //4x1 UV
            Float2  uv_value[4];
            Uchar2  uv_uc[4];
            in_pos[0] /= 2.0f;
            in_pos[1] = in_pos[2] / 2.0f;
            in_pos[2] = in_pos[4] / 2.0f;
            in_pos[3] = in_pos[6] / 2.0f;
            in_uv->read_interpolate_array<Float2, 4> (in_pos, uv_value);
            convert_to_uchar2_N<Float2, 4> (uv_value, uv_uc);
            out_uv->write_array_no_check<4> (x * 4, y, uv_uc);

            //2nd-line luma
            lut_pos[0].y = lut_pos[1].y = lut_pos[2].y = lut_pos[3].y = lut_pos[4].y = lut_pos[5].y =
                                              lut_pos[6].y = lut_pos[7].y = first.y + y_step;
            lut->read_interpolate_array<Float2, 8> (lut_pos, in_pos);
            in_luma->read_interpolate_array<float, 8> (in_pos, luma_value);
            convert_to_uchar_N<float, 8> (luma_value, luma_uc);
            out_luma->write_array_no_check<8> (out_x, out_y + 1, luma_uc);
        }
    return XCAM_RETURN_NO_ERROR;
}

}
