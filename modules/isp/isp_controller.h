/*
 * isp_controller.h - isp controller
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
 * Author: Wind Yuan <feng.yuan@intel.com>
 */
#ifndef XCAM_ISP_CONTROLLER_H
#define XCAM_ISP_CONTROLLER_H

#include "xcam_utils.h"
#include "x3a_isp_config.h"

namespace XCam {

class V4l2Device;
class X3aIspStatistics;
class X3aIspConfig;

class IspController {
public:
    explicit IspController (SmartPtr<V4l2Device> & device);
    ~IspController ();

    void init_sensor_mode_data (struct atomisp_sensor_mode_data *sensor_mode_data);
    XCamReturn get_sensor_mode_data (struct atomisp_sensor_mode_data &sensor_mode_data);
    XCamReturn get_isp_parameter (struct atomisp_parm &parameters);

    XCamReturn get_3a_statistics (SmartPtr<X3aIspStatistics> &stats);
    XCamReturn set_3a_config (X3aIspConfig *config);
    XCamReturn set_3a_exposure (X3aIspExposureResult *res);
    XCamReturn set_3a_exposure (const struct atomisp_exposure &exposure);
    XCamReturn set_3a_focus (const XCam3aResultFocus &focus);

private:

    XCAM_DEAD_COPY (IspController);

private:
    SmartPtr<V4l2Device> _device;
};

};

#endif //XCAM_ISP_CONTROLLER_H
