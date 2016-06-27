/*
 * isp_poll_thread.cpp - isp poll thread for event and buffer
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
 * Author: Yinhang Liu <yinhangx.liu@intel.com>
 */

#include "isp_poll_thread.h"
#include "x3a_statistics_queue.h"
#include <unistd.h>

namespace XCam {

class IspPollThread;

IspPollThread::IspPollThread ()
{
    XCAM_LOG_DEBUG ("IspPollThread constructed");
}

IspPollThread::~IspPollThread ()
{
    stop();

    XCAM_LOG_DEBUG ("~IspPollThread destructed");
}

bool
IspPollThread::set_isp_controller (SmartPtr<IspController>  &isp)
{
    XCAM_ASSERT (!_isp_controller.ptr());
    _isp_controller = isp;
    return true;
}

XCamReturn
IspPollThread::start ()
{
    _3a_stats_pool = new X3aStatisticsQueue;

    return PollThread::start ();
}

XCamReturn
IspPollThread::stop ()
{
    if (_3a_stats_pool.ptr ())
        _3a_stats_pool->stop ();

    return PollThread::stop ();
}

XCamReturn
IspPollThread::init_3a_stats_pool ()
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    struct atomisp_parm parameters;

    xcam_mem_clear (parameters);
    ret = _isp_controller->get_isp_parameter (parameters);
    if (ret != XCAM_RETURN_NO_ERROR ) {
        XCAM_LOG_WARNING ("get isp parameters failed");
        return ret;
    }
    if (!parameters.info.width || !parameters.info.height) {
        XCAM_LOG_WARNING ("get isp parameters width or height wrong");
        return XCAM_RETURN_ERROR_ISP;
    }
    _3a_stats_pool.dynamic_cast_ptr<X3aStatisticsQueue>()->set_grid_info (parameters.info);
    if (!_3a_stats_pool->reserve (6)) {
        XCAM_LOG_WARNING ("init_3a_stats_pool failed to reserve stats buffer.");
        return XCAM_RETURN_ERROR_MEM;
    }
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
IspPollThread::capture_3a_stats (SmartPtr<X3aStats> &stats)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    SmartPtr<X3aIspStatistics> new_stats =
        _3a_stats_pool->get_buffer (_3a_stats_pool).dynamic_cast_ptr<X3aIspStatistics> ();

    if (!new_stats.ptr()) {
        XCAM_LOG_WARNING ("request stats buffer failed.");
        return XCAM_RETURN_ERROR_MEM;
    }

    ret = _isp_controller->get_3a_statistics (new_stats);
    if (ret != XCAM_RETURN_NO_ERROR) {
        XCAM_LOG_WARNING ("get 3a stats from ISP failed");
        return ret;
    }

    if (!new_stats->fill_standard_stats ()) {
        XCAM_LOG_WARNING ("isp 3a stats failed to fill standard stats but continued");
    }

    stats = new_stats;
    return ret;
}


XCamReturn
IspPollThread::handle_events (struct v4l2_event &event)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    switch (event.type) {
    case V4L2_EVENT_ATOMISP_3A_STATS_READY:
        ret = handle_3a_stats_event (event);
        break;
    case V4L2_EVENT_FRAME_SYNC:
        break;
    default:
        ret = XCAM_RETURN_ERROR_UNKNOWN;
        break;
    }

    return ret;
}

};
