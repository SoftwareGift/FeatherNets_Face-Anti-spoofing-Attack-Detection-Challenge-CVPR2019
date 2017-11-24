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
};

class SoftGeoMapper
    : public SoftHandler, public GeoMapper
{
public:
    SoftGeoMapper (const char *name = "SoftGeoMap");
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

private:
    XCamReturn start_remap_task (const SmartPtr<ImageHandler::Parameters> &param);

private:
    SmartPtr<XCamSoftTasks::GeoMapTask>   _map_task;
    SmartPtr<Float2Image>                 _lookup_table;
};

extern SmartPtr<SoftHandler> create_soft_geo_mapper ();
}

#endif //XCAM_SOFT_GEO_MAP_H
