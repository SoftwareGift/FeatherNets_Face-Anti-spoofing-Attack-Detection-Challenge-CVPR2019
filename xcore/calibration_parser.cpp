/*
 * calibration_parser.cpp - parse fisheye calibration file
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

#include "calibration_parser.h"
#include "file_handle.h"

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

        XCAM_FAIL_RETURN (
            ERROR, intrinsic_param.poly_length <= XCAM_INTRINSIC_MAX_POLY_SIZE,
            XCAM_RETURN_ERROR_PARAM,
            "intrinsic poly length:%d is larger than max_size:%d.",
            intrinsic_param.poly_length, XCAM_INTRINSIC_MAX_POLY_SIZE);

        for(uint32_t i = 0; i < intrinsic_param.poly_length; i++) {
            tok_str = strtok_r(NULL, str_tokens, &tok_endptr);
            CHECK_NULL(tok_str);
            intrinsic_param.poly_coeff[i] = (strtof(tok_str, NULL));
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

XCamReturn
CalibrationParser::parse_intrinsic_file(const char *file_path, IntrinsicParameter &intrinsic_param)
{
    XCAM_ASSERT (file_path);

    FileHandle file_reader;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    std::vector<char> context;
    size_t file_size = 0;

    XCAM_FAIL_RETURN (
        WARNING, xcam_ret_is_ok (ret = file_reader.open (file_path, "r")), ret,
        "open intrinsic file(%s) failed.", file_path);
    XCAM_FAIL_RETURN (
        WARNING, xcam_ret_is_ok (ret = file_reader.get_file_size (file_size)), ret,
        "read intrinsic file(%s) failed to get file size.", file_path);
    context.resize (file_size + 1);
    XCAM_FAIL_RETURN (
        WARNING, xcam_ret_is_ok (ret = file_reader.read_file (&context[0], file_size)), ret,
        "read intrinsic file(%s) failed, file size:%d.", file_path, (int)file_size);
    file_reader.close ();
    context[file_size] = '\0';

    return parse_intrinsic_param (&context[0], intrinsic_param);
}

XCamReturn
CalibrationParser::parse_extrinsic_file(const char *file_path, ExtrinsicParameter &extrinsic_param)
{
    XCAM_ASSERT (file_path);

    FileHandle file_reader;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    std::vector<char> context;
    size_t file_size = 0;

    XCAM_FAIL_RETURN (
        WARNING, xcam_ret_is_ok (ret = file_reader.open (file_path, "r")), ret,
        "open extrinsic file(%s) failed.", file_path);
    XCAM_FAIL_RETURN (
        WARNING, xcam_ret_is_ok (ret = file_reader.get_file_size (file_size)), ret,
        "read extrinsic file(%s) failed to get file size.", file_path);
    context.resize (file_size + 1);
    XCAM_FAIL_RETURN (
        WARNING, xcam_ret_is_ok (ret = file_reader.read_file (&context[0], file_size)), ret,
        "read extrinsic file(%s) failed, file size:%d.", file_path, (int)file_size);
    file_reader.close ();
    context[file_size] = '\0';

    return parse_extrinsic_param (&context[0], extrinsic_param);
}

}
