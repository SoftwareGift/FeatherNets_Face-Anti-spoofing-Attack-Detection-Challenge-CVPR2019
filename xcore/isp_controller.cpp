/*
 * isp_controller.cpp - isp controller
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

#include "isp_controller.h"
#include "v4l2_device.h"
#include "x3a_statistics_queue.h"
#include "x3a_isp_config.h"

#include <linux/atomisp.h>

namespace XCam {

IspController::IspController (SmartPtr<V4l2Device> & device)
    : _device (device)
{
}
IspController::~IspController ()
{
}

XCamReturn
IspController::get_sensor_mode_data (struct atomisp_sensor_mode_data &sensor_mode_data)
{
    if ( _device->io_control (ATOMISP_IOC_G_SENSOR_MODE_DATA, &sensor_mode_data) < 0) {
        XCAM_LOG_WARNING (" get ISP sensor mode data failed");
        return XCAM_RETURN_ERROR_IOCTL;
    }
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
IspController::get_isp_parameter (struct atomisp_parm &parameters)
{
    if ( _device->io_control (ATOMISP_IOC_G_ISP_PARM, &parameters) < 0) {
        XCAM_LOG_WARNING (" get ISP parameters failed");
        return XCAM_RETURN_ERROR_IOCTL;
    }
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
IspController::get_3a_statistics (SmartPtr<X3aIspStatistics> &stats)
{
    struct atomisp_3a_statistics *isp_stats = NULL;

    XCAM_ASSERT (stats.ptr());
    XCAM_FAIL_RETURN (WARNING, stats.ptr(),
                      XCAM_RETURN_ERROR_PARAM, "stats empty");

    isp_stats =  stats->get_isp_stats ();

    if ( _device->io_control (ATOMISP_IOC_G_3A_STAT, isp_stats) < 0) {
        XCAM_LOG_WARNING (" get 3a stats failed from ISP");
        return XCAM_RETURN_ERROR_IOCTL;
    }
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
IspController::set_3a_config (X3aIspConfig *config)
{
    struct atomisp_parameters &isp_config = config->get_isp_configs ();
    if ( _device->io_control (ATOMISP_IOC_S_PARAMETERS, &isp_config) < 0) {
        XCAM_LOG_WARNING (" set 3a config failed to ISP");
        return XCAM_RETURN_ERROR_IOCTL;
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
IspController::set_3a_exposure (X3aIspExposureResult *res)
{
    const struct atomisp_exposure &exposure = res->get_isp_config ();
    return set_3a_exposure (exposure);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
IspController::set_3a_exposure (const struct atomisp_exposure &exposure)
{
    if ( _device->io_control (ATOMISP_IOC_S_EXPOSURE, (struct atomisp_exposure*)(&exposure)) < 0) {
        XCAM_LOG_WARNING (" set exposure result failed to device");
        return XCAM_RETURN_ERROR_IOCTL;
    }
    XCAM_LOG_DEBUG ("isp set exposure result, integration_time:%d, gain code:%d",
                    exposure.integration_time[0], exposure.gain[0]);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
IspController::set_3a_focus (const XCam3aResultFocus &focus)
{
    int position = focus.position;
    struct v4l2_control control;

    xcam_mem_clear (&control);
    control.id = V4L2_CID_FOCUS_ABSOLUTE;
    control.value = position;

    if (_device->io_control (VIDIOC_S_CTRL, &control) < 0) {
        XCAM_LOG_WARNING (" set focus result failed to device");
        return XCAM_RETURN_ERROR_IOCTL;
    }
    return XCAM_RETURN_NO_ERROR;
}


};
