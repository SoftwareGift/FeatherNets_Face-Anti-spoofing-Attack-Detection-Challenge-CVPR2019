/*
 * cv_utils.h - OpenCV Utilities
 *
 *  Copyright (c) 2019 Intel Corporation
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
 * Author: Yinhang Liu <yinhangx.liu@intel.com>
 */

#ifndef XCAM_CV_UTILS_H
#define XCAM_CV_UTILS_H

#include <interface/data_types.h>
#include <video_buffer.h>
#include "cv_std.h"

namespace XCam {

    bool convert_to_mat (const SmartPtr<VideoBuffer> &buffer, cv::Mat &img);
    bool convert_range_to_mat (const SmartPtr<VideoBuffer> &buffer, const Rect &range, cv::Mat &img);

    void write_image (
        const SmartPtr<VideoBuffer> &buf, const char *img_name, const char *frame_str, const char *idx_str);
    void write_image (
        const SmartPtr<VideoBuffer> &buf, const Rect &draw_rect,
        const char *img_name, const char *frame_str, const char *idx_str);

}

#endif // XCAM_CV_UTILS_H
