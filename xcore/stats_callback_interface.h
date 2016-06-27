/*
 * stats_callback_interface.h - statistics callback interface
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

#ifndef XCAM_STATS_CALLBACK_H
#define XCAM_STATS_CALLBACK_H

#include "xcam_utils.h"
#include "xcam_mutex.h"


namespace XCam {

class X3aStats;
class BufferProxy;

class StatsCallback {
public:
    StatsCallback () {}
    virtual ~StatsCallback() {}
    virtual XCamReturn x3a_stats_ready (const SmartPtr<X3aStats> &stats) {
        XCAM_UNUSED (stats);
        return XCAM_RETURN_NO_ERROR;
    }
    virtual XCamReturn dvs_stats_ready () {
        return XCAM_RETURN_NO_ERROR;
    }
    virtual XCamReturn scaled_image_ready (const SmartPtr<BufferProxy> &buffer) = 0;

private:
    XCAM_DEAD_COPY (StatsCallback);
};

}
#endif //XCAM_STATS_CALLBACK_H
