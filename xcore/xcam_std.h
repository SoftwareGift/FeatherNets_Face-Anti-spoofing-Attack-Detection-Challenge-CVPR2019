/*
 * xcam_std.h - xcam std
 *
 *  Copyright (c) 2014-2017 Intel Corporation
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

#ifndef XCAM_STD_H
#define XCAM_STD_H

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <base/xcam_common.h>
#include <xcam_obj_debug.h>
extern "C" {
#include <linux/videodev2.h>
}
#include <cinttypes>
#include <vector>
#include <smartptr.h>

namespace XCam {

static const int64_t InvalidTimestamp = INT64_C(-1);

enum NV12PlaneIdx {
    NV12PlaneYIdx = 0,
    NV12PlaneUVIdx,
    NV12PlaneMax,
};

};

#endif //XCAM_STD_H