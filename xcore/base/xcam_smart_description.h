/*
 * xcam_smart_description.h - libxcam smart analysis description
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
 * Author: Zong Wei <wei.zong@intel.com>
 */

#ifndef C_XCAM_SMART_ANALYSIS_DESCRIPTION_H
#define C_XCAM_SMART_ANALYSIS_DESCRIPTION_H

#include <base/xcam_common.h>
#include <base/xcam_params.h>
#include <base/xcam_3a_result.h>
#include <base/xcam_3a_stats.h>

XCAM_BEGIN_DECLARE

#define XCAM_SMART_ANALYSIS_LIB_DESCRIPTION "xcam_smart_analysis_desciption"

typedef struct _XCamSmartAnalysisContext XCamSmartAnalysisContext;

/* C interface of Smart Analysis lib */
typedef struct _XCamSmartAnalysisDescription {
    uint32_t                        version;
    uint32_t                        size;
    XCamReturn (*create_context)  (XCamSmartAnalysisContext **context);
    XCamReturn (*destroy_context) (XCamSmartAnalysisContext *context);
    XCamReturn (*update_params)   (XCamSmartAnalysisContext *context, XCamSmartAnalysisParam *params);
    XCamReturn (*analyze)         (XCamSmartAnalysisContext *context, XCamVideoBuffer *buffer);
    XCamReturn (*get_results)     (XCamSmartAnalysisContext *context, XCam3aResultHead *results[], uint32_t *res_count);
    void       (*free_results)    (XCam3aResultHead *results[], uint32_t res_count);
} XCamSmartAnalysisDescription;

XCAM_END_DECLARE

#endif //C_XCAM_SMART_ANALYSIS_DESCRIPTION_H
