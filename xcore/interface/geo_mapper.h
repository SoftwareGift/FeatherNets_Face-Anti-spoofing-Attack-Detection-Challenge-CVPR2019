/*
 * geo_mapper.h - geometry mapper interface
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

#ifndef XCAM_INTERFACE_GEO_MAPPER_H
#define XCAM_INTERFACE_GEO_MAPPER_H

#include <xcam_std.h>
#include <video_buffer.h>
#include <interface/data_types.h>

namespace XCam {

class GeoMapper
{
public:
    GeoMapper ();
    virtual ~GeoMapper ();
    static SmartPtr<GeoMapper> create_ocl_geo_mapper ();
    static SmartPtr<GeoMapper> create_soft_geo_mapper ();

    //2D table
    virtual bool set_lookup_table (const PointFloat2 *data, uint32_t width, uint32_t height) = 0;
    bool set_factors (float x, float y);
    void get_factors (float &x, float &y) const {
        x = _factor_x;
        y = _factor_y;
    }
    bool set_output_size (uint32_t width, uint32_t height);
    void get_output_size (uint32_t &width, uint32_t &height) const {
        width = _out_width;
        height = _out_height;
    }

    virtual XCamReturn remap (
        const SmartPtr<VideoBuffer> &in,
        SmartPtr<VideoBuffer> &out_buf) = 0;

protected:
    bool auto_calculate_factors (uint32_t lut_w, uint32_t lut_h);

private:
    uint32_t     _out_width, _out_height;
    float        _factor_x, _factor_y;
};

}
#endif //XCAM_INTERFACE_GEO_MAPPER_H
