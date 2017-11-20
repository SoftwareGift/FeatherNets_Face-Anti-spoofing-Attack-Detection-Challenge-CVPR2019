/*
 * geo_mapper.cpp - geometry mapper implementation
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

#include "geo_mapper.h"

namespace XCam {

GeoMapper::GeoMapper ()
    : _out_width (0)
    , _out_height (0)
    , _factor_x (0.0f)
    , _factor_y (0.0f)
{}

GeoMapper::~GeoMapper ()
{
}

bool
GeoMapper::set_factors (float x, float y)
{
    XCAM_FAIL_RETURN (
        ERROR, !XCAM_DOUBLE_EQUAL_AROUND (x, 0.0f) && !XCAM_DOUBLE_EQUAL_AROUND (y, 0.0f), false,
        "GeoMapper set factors failed. (x:%.3f, h:%.3f)", x, y);
    _factor_x = x;
    _factor_y = y;

    return true;
}

bool
GeoMapper::set_output_size (uint32_t width, uint32_t height)
{
    XCAM_FAIL_RETURN (
        ERROR, width && height, false,
        "GeoMapper set output size failed. (w:%d, h:%d)",
        width, height);

    _out_width = width;
    _out_height = height;
    return true;
}

bool
GeoMapper::auto_calculate_factors (uint32_t lut_w, uint32_t lut_h)
{
    XCAM_FAIL_RETURN (
        ERROR, _out_width > 1 && _out_height > 1, false,
        "GeoMapper auto calculate factors failed. output size was not set. (w:%d, h:%d)",
        _out_width, _out_height);
    XCAM_FAIL_RETURN (
        ERROR, lut_w > 1 && lut_w > 1, false,
        "GeoMapper auto calculate factors failed. lookuptable size need > 1. but set with (w:%d, h:%d)",
        lut_w, lut_h);

    XCAM_ASSERT (lut_w && lut_h);
    _factor_x = (_out_width - 1.0f) / (lut_w - 1.0f);
    _factor_y = (_out_height - 1.0f) / (lut_h - 1.0f);
    return true;
}

}
