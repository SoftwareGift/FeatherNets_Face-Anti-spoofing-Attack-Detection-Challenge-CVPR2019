/*
 * isp_poll_thread.h - isp poll thread for event and buffer
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

#ifndef XCAM_ISP_POLL_THREAD_H
#define XCAM_ISP_POLL_THREAD_H

#include "poll_thread.h"
#include "isp_controller.h"

namespace XCam {

class IspPollThread
    : public PollThread
{
public:
    explicit IspPollThread ();
    virtual ~IspPollThread ();

    bool set_isp_controller (SmartPtr<IspController> &isp);

    virtual XCamReturn start();
    virtual XCamReturn stop ();

protected:
    virtual XCamReturn handle_events (struct v4l2_event &event);

private:
    virtual XCamReturn init_3a_stats_pool ();
    virtual XCamReturn capture_3a_stats (SmartPtr<X3aStats> &stats);

private:
    XCAM_DEAD_COPY (IspPollThread);

private:
    SmartPtr<X3aStatsPool>           _3a_stats_pool;
    SmartPtr<IspController>          _isp_controller;
};

};

#endif // XCAM_ISP_POLL_THREAD_H
