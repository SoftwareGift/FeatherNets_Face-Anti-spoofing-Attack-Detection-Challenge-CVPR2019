/*
 * fake_v4l2_device.h - fake v4l2 device
 *
 *  Copyright (c) 2014-2015 Intel Corporation
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
 * Author: Jia Meng <jia.meng@intel.com>
 */

#ifndef XCAM_FAKE_V4L2_DEVICE_H
#define XCAM_FAKE_V4L2_DEVICE_H

#include "v4l2_device.h"
#include <linux/atomisp.h>

namespace XCam {

class FakeV4l2Device
    : public V4l2Device
{
public:
    FakeV4l2Device ()
        : V4l2Device ("/dev/null")
    {}

    int io_control (int cmd, void *arg)
    {
        int ret = 0;

        switch (cmd) {
        case ATOMISP_IOC_G_SENSOR_MODE_DATA: {
            struct atomisp_sensor_mode_data *sensor_mode_data = (struct atomisp_sensor_mode_data *)arg;
            sensor_mode_data->coarse_integration_time_min = 1;
            sensor_mode_data->coarse_integration_time_max_margin = 1;
            sensor_mode_data->fine_integration_time_min = 0;
            sensor_mode_data->fine_integration_time_max_margin = 0;
            sensor_mode_data->fine_integration_time_def = 0;
            sensor_mode_data->frame_length_lines = 1125;
            sensor_mode_data->line_length_pck = 1320;
            sensor_mode_data->read_mode = 0;
            sensor_mode_data->vt_pix_clk_freq_mhz = 37125000;
            sensor_mode_data->crop_horizontal_start = 0;
            sensor_mode_data->crop_vertical_start = 0;
            sensor_mode_data->crop_horizontal_end = 1920;
            sensor_mode_data->crop_vertical_end = 1080;
            sensor_mode_data->output_width = 1920;
            sensor_mode_data->output_height = 1080;
            sensor_mode_data->binning_factor_x = 1;
            sensor_mode_data->binning_factor_y = 1;
            break;
        }
        case VIDIOC_ENUM_FMT:
            ret = -1;
            break;
        default:
            break;
        }
        return ret;
    }
};

};
#endif // XCAM_FAKE_V4L2_DEVICE_H
