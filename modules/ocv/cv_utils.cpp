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

const static cv::Scalar color = cv::Scalar (0, 0, 255);
const static int fontFace = cv::FONT_HERSHEY_COMPLEX;

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
    // buffer->unmap ();

    return true;
}

void write_image (
    const SmartPtr<VideoBuffer> &buf, const char *img_name, const char *frame_str, const char *idx_str)
{
    XCAM_ASSERT (img_name);

    cv::Mat mat;
    convert_to_mat (buf, mat);

    if(frame_str)
        cv::putText (mat, frame_str, cv::Point(20, 50), fontFace, 2.0, color, 2, 8, false);
    if(idx_str)
        cv::putText (mat, idx_str, cv::Point(20, 110), fontFace, 2.0, color, 2, 8, false);

    cv::imwrite (img_name, mat);
}

void write_image (
    const SmartPtr<VideoBuffer> &buf, const Rect &draw_rect,
    const char *img_name, const char *frame_str, const char *idx_str)
{
    XCAM_ASSERT (img_name && frame_str && idx_str);

    cv::Mat mat;
    convert_to_mat (buf, mat);

    const Rect &rect = draw_rect;
    cv::putText (mat, frame_str, cv::Point(rect.pos_x, 30), fontFace, 0.8f, color, 2, 8, false);
    cv::putText (mat, idx_str, cv::Point(rect.pos_x, 70), fontFace, 0.8f, color, 2, 8, false);

    cv::line (mat, cv::Point(rect.pos_x, rect.pos_y), cv::Point(rect.pos_x + rect.width, rect.pos_y), color, 1);
    cv::line (mat, cv::Point(rect.pos_x, rect.pos_y + rect.height),
              cv::Point(rect.pos_x + rect.width, rect.pos_y + rect.height), color, 1);

    VideoBufferInfo info = buf->get_video_info ();
    cv::line (mat, cv::Point(rect.pos_x, 0), cv::Point(rect.pos_x, info.height), color, 2);
    cv::line (mat, cv::Point(rect.pos_x + rect.width, 0), cv::Point(rect.pos_x + rect.width, info.height), color, 2);

    cv::imwrite (img_name, mat);
}

}
