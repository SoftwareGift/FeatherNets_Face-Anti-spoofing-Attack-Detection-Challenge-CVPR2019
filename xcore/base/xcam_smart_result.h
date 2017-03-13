/*
 * xcam_smart_result.h - smart result(meta data)
 *
 *  Copyright (c) 2016-2017 Intel Corporation
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

#ifndef C_XCAM_SMART_RESULT_H
#define C_XCAM_SMART_RESULT_H

#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <base/xcam_3a_result.h>

XCAM_BEGIN_DECLARE

typedef struct _XCamFaceInfo {
    uint32_t id;
    uint32_t pos_x;
    uint32_t pos_y;
    uint32_t width;
    uint32_t height;
    uint32_t factor;
    uint32_t landmark[128];
} XCamFaceInfo;

/*
 * Face detection result
 * head.type = XCAM_3A_RESULT_FACE_DETECTION;
 * head.process_type = XCAM_IMAGE_PROCESS_POST;
 * head.destroy = free fd result.
 */

typedef struct _XCamFDResult {
    XCam3aResultHead      head;
    uint32_t              face_num;
    XCamFaceInfo          faces[0];
} XCamFDResult;

/*
 * Digital Video Stabilizer result
 * head.type = XCAM_3A_RESULT_DVS;
 * head.process_type = XCAM_IMAGE_PROCESS_POST;
 * head.destroy = free dvs result.
 */

typedef struct _XCamDVSResult {
    XCam3aResultHead      head;
    int                   frame_id;
    int                   frame_width;
    int                   frame_height;
    double                proj_mat[9];
} XCamDVSResult;

XCAM_END_DECLARE

#endif //C_XCAM_SMART_RESULT_H

