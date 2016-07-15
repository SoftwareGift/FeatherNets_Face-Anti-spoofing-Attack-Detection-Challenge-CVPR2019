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

#ifndef XCAM_SENSOR_DESCRIPTOR_H
#define XCAM_SENSOR_DESCRIPTOR_H

#include "xcam_utils.h"
#include <linux/atomisp.h>

namespace XCam {

class SensorDescriptor {
public:
    explicit SensorDescriptor ();
    virtual ~SensorDescriptor ();

    void set_sensor_data (struct atomisp_sensor_mode_data &data);
    virtual bool is_ready ();

    // Input: exposure_time
    // Output: coarse_time, fine_time
    virtual bool exposure_time_to_integration (
        int32_t exposure_time, uint32_t &coarse_time, uint32_t &fine_time);
    // Input: coarse_time, fine_time
    // Output: exposure_time
    virtual bool exposure_integration_to_time (
        uint32_t coarse_time, uint32_t fine_time, int32_t &exposure_time);

    // Input : analog_gain, digital_gain
    // Output: analog_code, digital_code
    virtual bool exposure_gain_to_code (
        double analog_gain, double digital_gain,
        int32_t &analog_code, int32_t &digital_code);

    // Input : analog_code, digital_code
    // Output : analog_gain, digital_gain
    virtual bool exposure_code_to_gain (
        int32_t analog_code, int32_t digital_code,
        double &analog_gain, double &digital_gain);

private:
    XCAM_DEAD_COPY (SensorDescriptor);

private:
    struct atomisp_sensor_mode_data _sensor_data;
};

};
#endif //XCAM_SENSOR_DESCRIPTOR_H
