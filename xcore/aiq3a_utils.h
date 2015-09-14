/*
 * aiq3a_util.h - aiq 3a utility:
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
 * Author: Shincy Tu <shincy.tu@intel.com>
 */

#ifndef XCAM_AIQ_UTILS_H
#define XCAM_AIQ_UTILS_H


#include "xcam_utils.h"
#include "x3a_result.h"

#include <base/xcam_3a_stats.h>
#include <linux/atomisp.h>

namespace XCam {
bool translate_3a_stats (XCam3AStats *from, struct atomisp_3a_statistics *to);
uint32_t translate_3a_results_to_xcam (XCam::X3aResultList &list,
                                       XCam3aResultHead *results[], uint32_t max_count);

void free_3a_result (XCam3aResultHead *result);
}

#endif //XCAM_AIQ_UTILS_H
