/*
 * cv_calibration_parser.cpp - parse fisheye calibration file
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

#include "cv_calibration_parser.h"

namespace XCam {

CalibrationParser::CalibrationParser()
{
}

#define CHECK_NULL(ptr) \
    if(ptr == NULL) { \
        XCAM_LOG_ERROR("Parse file failed"); \
        return XCAM_RETURN_ERROR_FILE; \
    }

XCamReturn
CalibrationParser::parse_intrinsic_param(char *file_body, IntrinsicParameter &intrinsic_param)
{
    char *line_str = NULL;
    char *line_endptr = NULL;
    char *tok_str = NULL;
    char *tok_endptr = NULL;
    static const char *line_tokens = "\r\n";
    static const char *str_tokens = " \t";

    do {
        line_str = strtok_r(file_body, line_tokens, &line_endptr);
        CHECK_NULL(line_str);
        tok_str = strtok_r(line_str, str_tokens, &tok_endptr);
        while(tok_str == NULL || tok_str[0] == '#') {
            line_str = strtok_r(NULL, line_tokens, &line_endptr);
            CHECK_NULL(line_str);
            tok_str = strtok_r(line_str, str_tokens, &tok_endptr);
        }
    
        line_str = strtok_r(NULL, line_tokens, &line_endptr);
        CHECK_NULL(line_str);
        tok_str = strtok_r(line_str, str_tokens, &tok_endptr);
        while(tok_str == NULL || tok_str[0] == '#') {
            line_str = strtok_r(NULL, line_tokens, &line_endptr);
            CHECK_NULL(line_str);
            tok_str = strtok_r(line_str, str_tokens, &tok_endptr);
        }
        intrinsic_param.poly_length = strtol(tok_str, NULL, 10);

        for(int i = 0; i < intrinsic_param.poly_length; i++) {
            tok_str = strtok_r(NULL, str_tokens, &tok_endptr);
            CHECK_NULL(tok_str);
            intrinsic_param.poly_coeff.push_back(strtof(tok_str, NULL));
        }

        line_str = strtok_r(NULL, line_tokens, &line_endptr);
        CHECK_NULL(line_str);
        tok_str = strtok_r(line_str, str_tokens, &tok_endptr);
        while(tok_str == NULL || tok_str[0] == '#') {
            line_str = strtok_r(NULL, line_tokens, &line_endptr);
            CHECK_NULL(line_str);
            tok_str = strtok_r(line_str, str_tokens, &tok_endptr);
        }
        intrinsic_param.yc = strtof(tok_str, NULL);

        tok_str = strtok_r(NULL, str_tokens, &tok_endptr);
        CHECK_NULL(tok_str);
        intrinsic_param.xc = strtof(tok_str, NULL);

        line_str = strtok_r(NULL, line_tokens, &line_endptr);
        CHECK_NULL(line_str);
        tok_str = strtok_r(line_str, str_tokens, &tok_endptr);
        while(tok_str == NULL || tok_str[0] == '#') {
            line_str = strtok_r(NULL, line_tokens, &line_endptr);
            CHECK_NULL(line_str);
            tok_str = strtok_r(line_str, str_tokens, &tok_endptr);
        }
        intrinsic_param.c = strtof(tok_str, NULL);

        tok_str = strtok_r(NULL, str_tokens, &tok_endptr);
        CHECK_NULL(tok_str);
        intrinsic_param.d = strtof(tok_str, NULL);

        tok_str = strtok_r(NULL, str_tokens, &tok_endptr);
        CHECK_NULL(tok_str);
        intrinsic_param.e = strtof(tok_str, NULL);
    } while(0);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CalibrationParser::parse_extrinsic_param(char *file_body, ExtrinsicParameter &extrinsic_param)
{
    char *line_str = NULL;
    char *line_endptr = NULL;
    char *tok_str = NULL;
    char *tok_endptr = NULL;
    static const char *line_tokens = "\r\n";
    static const char *str_tokens = " \t";

    do {
        line_str = strtok_r(file_body, line_tokens, &line_endptr);
        CHECK_NULL(line_str);
        tok_str = strtok_r(line_str, str_tokens, &tok_endptr);
        while(tok_str == NULL || tok_str[0] == '#') {
            line_str = strtok_r(NULL, line_tokens, &line_endptr);
            CHECK_NULL(line_str);
            tok_str = strtok_r(line_str, str_tokens, &tok_endptr);
        }
        extrinsic_param.trans_x = strtof(tok_str, NULL);

        line_str = strtok_r(NULL, line_tokens, &line_endptr);
        CHECK_NULL(line_str);
        tok_str = strtok_r(line_str, str_tokens, &tok_endptr);
        while(tok_str == NULL || tok_str[0] == '#') {
            line_str = strtok_r(NULL, line_tokens, &line_endptr);
            CHECK_NULL(line_str);
            tok_str = strtok_r(line_str, str_tokens, &tok_endptr);
        }
        extrinsic_param.trans_y = strtof(tok_str, NULL);

        line_str = strtok_r(NULL, line_tokens, &line_endptr);
        CHECK_NULL(line_str);
        tok_str = strtok_r(line_str, str_tokens, &tok_endptr);
        while(tok_str == NULL || tok_str[0] == '#') {
            line_str = strtok_r(NULL, line_tokens, &line_endptr);
            CHECK_NULL(line_str);
            tok_str = strtok_r(line_str, str_tokens, &tok_endptr);
        }
        extrinsic_param.trans_z = strtof(tok_str, NULL);

        line_str = strtok_r(NULL, line_tokens, &line_endptr);
        CHECK_NULL(line_str);
        tok_str = strtok_r(line_str, str_tokens, &tok_endptr);
        while(tok_str == NULL || tok_str[0] == '#') {
            line_str = strtok_r(NULL, line_tokens, &line_endptr);
            CHECK_NULL(line_str);
            tok_str = strtok_r(line_str, str_tokens, &tok_endptr);
        }
        extrinsic_param.roll = strtof(tok_str, NULL);

        line_str = strtok_r(NULL, line_tokens, &line_endptr);
        CHECK_NULL(line_str);
        tok_str = strtok_r(line_str, str_tokens, &tok_endptr);
        while(tok_str == NULL || tok_str[0] == '#') {
            line_str = strtok_r(NULL, line_tokens, &line_endptr);
            CHECK_NULL(line_str);
            tok_str = strtok_r(line_str, str_tokens, &tok_endptr);
        }
        extrinsic_param.pitch = strtof(tok_str, NULL);

        line_str = strtok_r(NULL, line_tokens, &line_endptr);
        CHECK_NULL(line_str);
        tok_str = strtok_r(line_str, str_tokens, &tok_endptr);
        while(tok_str == NULL || tok_str[0] == '#') {
            line_str = strtok_r(NULL, line_tokens, &line_endptr);
            CHECK_NULL(line_str);
            tok_str = strtok_r(line_str, str_tokens, &tok_endptr);
        }
        extrinsic_param.yaw = strtof(tok_str, NULL);
     } while(0);

    return XCAM_RETURN_NO_ERROR;
}

}
