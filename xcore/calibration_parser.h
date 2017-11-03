/*
 * calibration_parser.h - parse fisheye calibration file
 *
 *  Copyright (c) 2016-2017 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Author: Junkai Wu <junkai.wu@intel.com>
 */

#ifndef XCAM_CALIBRATION_PARSER_H
#define XCAM_CALIBRATION_PARSER_H

#include "xcam_utils.h"
#include <vector>

namespace XCam {

struct IntrinsicParameter {
    float xc;
    float yc;
    float c;
    float d;
    float e;
    uint32_t poly_length;

    std::vector<float> poly_coeff;
};

struct ExtrinsicParameter {
    float trans_x;
    float trans_y;
    float trans_z;
    float roll;
    float pitch;
    float yaw;
};

class CalibrationParser
{

public:
    explicit CalibrationParser ();

    XCamReturn parse_intrinsic_param(char *file_body, IntrinsicParameter &intrinsic_param);
    XCamReturn parse_extrinsic_param(char *file_body, ExtrinsicParameter &extrinsic_param);

private:
    XCAM_DEAD_COPY (CalibrationParser);
};

}

#endif // XCAM_CALIBRATION_PARSER_H
