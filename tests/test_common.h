/*
 * test_utils.h - test utils
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
 * Author: John Ye <john.ye@intel.com>
 */

#ifndef XCAM_TEST_COMMON_H
#define XCAM_TEST_COMMON_H

#undef CHECK_DECLARE
#undef CHECK
#undef CHECK_CONTINUE

#define CHECK_DECLARE(level, ret, statement, msg, ...) \
    if ((ret) != XCAM_RETURN_NO_ERROR) {        \
        XCAM_LOG_##level (msg, ## __VA_ARGS__);   \
        statement;                              \
    }

#define CHECK(ret, msg, ...)  \
    CHECK_DECLARE(ERROR, ret, return -1, msg, ## __VA_ARGS__)

#define CHECK_CONTINUE(ret, msg, ...)  \
    CHECK_DECLARE(WARNING, ret, , msg, ## __VA_ARGS__)

#define CAPTURE_DEVICE_VIDEO "/dev/video3"
#define CAPTURE_DEVICE_STILL "/dev/video0"
#define DEFAULT_CAPTURE_DEVICE CAPTURE_DEVICE_VIDEO

#define DEFAULT_EVENT_DEVICE   "/dev/v4l-subdev6"
#define DEFAULT_CPF_FILE       "/etc/atomisp/imx185.cpf"
#define DEFAULT_SAVE_FILE_NAME "capture_buffer.raw"

#endif  // XCAM_TEST_COMMON_H
