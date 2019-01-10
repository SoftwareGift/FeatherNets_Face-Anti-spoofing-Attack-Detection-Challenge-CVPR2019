/*
 * cv_utils.cpp - OpenCV Utilities
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

#include "cv_utils.h"

namespace XCam {

bool convert_to_mat (const SmartPtr<VideoBuffer> &buffer, cv::Mat &img)
{
    VideoBufferInfo info = buffer->get_video_info ();
    XCAM_FAIL_RETURN (ERROR, info.format == V4L2_PIX_FMT_NV12, false, "convert_to_mat only support NV12 format");

    uint8_t *mem = buffer->map ();
    XCAM_FAIL_RETURN (ERROR, mem, false, "convert_to_mat buffer map failed");

    cv::Mat mat = cv::Mat (info.aligned_height * 3 / 2, info.width, CV_8UC1, mem, info.strides[0]);
    cv::cvtColor (mat, img, cv::COLOR_YUV2BGR_NV12);
    buffer->unmap ();

    return true;
}

bool convert_range_to_mat (const SmartPtr<VideoBuffer> &buffer, const Rect &range, cv::Mat &img)
{
    VideoBufferInfo info = buffer->get_video_info ();

    uint8_t *mem = buffer->map ();
    XCAM_FAIL_RETURN (ERROR, mem, false, "convert_range_to_mat buffer map failed");

    uint8_t *start = mem + range.pos_y * info.strides[0] + range.pos_x;
    img = cv::Mat (range.height, range.width, CV_8UC1, start, info.strides[0]);
    buffer->unmap ();

    return true;
}

}
