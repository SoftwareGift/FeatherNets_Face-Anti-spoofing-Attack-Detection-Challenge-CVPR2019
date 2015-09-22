/*
 * sensor_descriptor.h - sensor descriptor
 *
 *  Copyright (c) 2015 Intel Corporation
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

#include "sensor_descriptor.h"
#include <math.h>

namespace XCam {

SensorDescriptor::SensorDescriptor ()
{
    xcam_mem_clear (_sensor_data);
}

SensorDescriptor::~SensorDescriptor ()
{
}

bool
SensorDescriptor::is_ready ()
{
    return (_sensor_data.line_length_pck > 0);
}

void
SensorDescriptor::set_sensor_data (struct atomisp_sensor_mode_data &data)
{
    _sensor_data = data;
}

bool
SensorDescriptor::exposure_time_to_integration (
    int32_t exposure_time, uint32_t &coarse_time, uint32_t &fine_time)
{
    if (exposure_time < 0 || !is_ready ())
        return false;

    uint32_t pixel_periods =  ((uint64_t)exposure_time) * _sensor_data.vt_pix_clk_freq_mhz / XCAM_SECONDS_2_TIMESTAMP (1);

    coarse_time = pixel_periods / _sensor_data.line_length_pck;
    fine_time = pixel_periods % _sensor_data.line_length_pck;
    return true;
}

bool
SensorDescriptor::exposure_integration_to_time (
    uint32_t coarse_time, uint32_t fine_time, int32_t &exposure_time)
{
    if (!is_ready ())
        return false;

    uint64_t pixel_periods = coarse_time * _sensor_data.line_length_pck + fine_time;
    exposure_time = pixel_periods * XCAM_SECONDS_2_TIMESTAMP(1) / _sensor_data.vt_pix_clk_freq_mhz;
    return true;
}

bool
SensorDescriptor::exposure_gain_to_code (
    double analog_gain, double digital_gain,
    int32_t &analog_code, int32_t &digital_code)
{
    XCAM_ASSERT (digital_gain == 1.0);
    double db = log10 (analog_gain * digital_gain) * 20;
    if (db > 48)
        db = 48;
    analog_code =  (uint32_t) (db * 160.0 / 48);
    digital_code = 0;
    return true;
}

bool
SensorDescriptor::exposure_code_to_gain (
    int32_t analog_code, int32_t digital_code,
    double &analog_gain, double &digital_gain)
{
    XCAM_UNUSED (digital_code);
    double db = analog_code * 48.0 / 160.0;
    analog_gain = pow (10.0, db / 20.0);
    digital_gain = 1.0;

    return true;
}

};
