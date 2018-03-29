/*
 * soft_geo_mapper.h - soft geometry map class
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

#ifndef XCAM_SOFT_GEO_MAP_H
#define XCAM_SOFT_GEO_MAP_H

#include <xcam_std.h>
#include <interface/geo_mapper.h>
#include <soft/soft_handler.h>
#include <soft/soft_image.h>

namespace XCam {

namespace XCamSoftTasks {
class GeoMapTask;
class GeoMapDualConstTask;
class GeoMapDualCurveTask;
};

class SoftGeoMapper
    : public SoftHandler, public GeoMapper
{
public:
    SoftGeoMapper (const char *name = "SoftGeoMapper");
    ~SoftGeoMapper ();

    bool set_lookup_table (const PointFloat2 *data, uint32_t width, uint32_t height);

    //derived from SoftHandler
    virtual XCamReturn terminate ();

    void remap_task_done (
        const SmartPtr<Worker> &worker, const SmartPtr<Worker::Arguments> &args, const XCamReturn error);

protected:
    //derived from interface
    XCamReturn remap (
        const SmartPtr<VideoBuffer> &in,
        SmartPtr<VideoBuffer> &out_buf);

    //derived from SoftHandler
    XCamReturn configure_resource (const SmartPtr<Parameters> &param);
    XCamReturn start_work (const SmartPtr<Parameters> &param);

    void set_work_size (uint32_t thread_x, uint32_t thread_y, uint32_t luma_width, uint32_t luma_height);
    SmartPtr<XCamSoftTasks::GeoMapTask> &get_map_task () {
        return _map_task;
    }
    SmartPtr<Float2Image> &get_lookup_table () {
        return _lookup_table;
    }

protected:
    virtual bool init_factors ();
    virtual SmartPtr<XCamSoftTasks::GeoMapTask> create_remap_task ();
    virtual XCamReturn start_remap_task (const SmartPtr<ImageHandler::Parameters> &param);

private:
    SmartPtr<XCamSoftTasks::GeoMapTask>   _map_task;
    SmartPtr<Float2Image>                 _lookup_table;
};

extern SmartPtr<SoftHandler> create_soft_geo_mapper ();

class SoftDualConstGeoMapper
    : public SoftGeoMapper
{
public:
    SoftDualConstGeoMapper (const char *name = "SoftDualConstGeoMapper");
    ~SoftDualConstGeoMapper ();

    bool set_left_factors (float x, float y);
    void get_left_factors (float &x, float &y) {
        x = _left_factor_x;
        y = _left_factor_y;
    }
    bool set_right_factors (float x, float y);
    void get_right_factors (float &x, float &y) {
        x = _right_factor_x;
        y = _right_factor_y;
    }

    virtual void remap_task_done (
        const SmartPtr<Worker> &worker, const SmartPtr<Worker::Arguments> &args, const XCamReturn error);

protected:
    XCamReturn prepare_arguments (const SmartPtr<Worker::Arguments> &args,
        const SmartPtr<ImageHandler::Parameters> &param);
    virtual bool auto_calculate_factors (uint32_t lut_w, uint32_t lut_h);

protected:
    virtual bool init_factors ();
    virtual SmartPtr<XCamSoftTasks::GeoMapTask> create_remap_task ();
    virtual XCamReturn start_remap_task (const SmartPtr<ImageHandler::Parameters> &param);

private:
    float        _left_factor_x, _left_factor_y;
    float        _right_factor_x, _right_factor_y;
};

class SoftDualCurveGeoMapper
    : public SoftDualConstGeoMapper
{
public:
    SoftDualCurveGeoMapper (const char *name = "SoftDualCurveGeoMapper");
    ~SoftDualCurveGeoMapper ();

    void set_scaled_height (float scaled_height) {
        _scaled_height = scaled_height;
    }

    virtual void remap_task_done (
        const SmartPtr<Worker> &worker, const SmartPtr<Worker::Arguments> &args, const XCamReturn error);

private:
    virtual SmartPtr<XCamSoftTasks::GeoMapTask> create_remap_task ();
    virtual XCamReturn start_remap_task (const SmartPtr<ImageHandler::Parameters> &param);

private:
    float        _scaled_height;
};

}
#endif //XCAM_SOFT_GEO_MAP_H
