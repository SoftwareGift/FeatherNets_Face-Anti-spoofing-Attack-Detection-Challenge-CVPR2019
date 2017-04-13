/*
 * cv_feature_match.cpp - optical flow feature match
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
 * Author: Wind Yuan <feng.yuan@intel.com>
 * Author: Yinhang Liu <yinhangx.liu@intel.com>
 */

#include "cv_feature_match.h"

using namespace XCam;

#define XCAM_OF_DEBUG 0
#define XCAM_OF_DRAW_SCALE 2

static const int sitch_min_width = 56;

static const int min_corners = 8;
static const float offset_factor = 0.8f;

static const int delta_count = 4;  // cur_count - last_count
static const float delta_mean_offset = 1.0f; //0.1f;  // cur_mean_offset - last_mean_offset
static const float delta_offset = 12.0f;  // cur_mean_offset - last_offset

void
init_opencv_ocl (SmartPtr<CLContext> context)
{
    static bool is_ocl_inited = false;

    if (!is_ocl_inited) {
        cl_platform_id platform_id = CLDevice::instance()->get_platform_id ();
        char *platform_name = CLDevice::instance()->get_platform_name ();
        cl_device_id device_id = CLDevice::instance()->get_device_id ();
        cl_context context_id = context->get_context_id ();
        cv::ocl::attachContext (platform_name, platform_id, context_id, device_id);

        is_ocl_inited = true;
    }
}

bool
convert_to_mat (SmartPtr<CLContext> context, SmartPtr<DrmBoBuffer> buffer, cv::Mat &image)
{
    SmartPtr<CLBuffer> cl_buffer = new CLVaBuffer (context, buffer);
    VideoBufferInfo info = buffer->get_video_info ();
    cl_mem cl_mem_id = cl_buffer->get_mem_id ();

    cv::UMat umat;
    cv::ocl::convertFromBuffer (cl_mem_id, info.strides[0], info.height * 3 / 2, info.width, CV_8U, umat);
    if (umat.empty ()) {
        XCAM_LOG_ERROR ("convert bo buffer to UMat failed");
        return false;
    }

    cv::Mat mat;
    umat.copyTo (mat);
    if (mat.empty ()) {
        XCAM_LOG_ERROR ("copy UMat to Mat failed");
        return false;
    }

    cv::cvtColor (mat, image, cv::COLOR_YUV2BGR_NV12);
    return true;
}

bool
convert_to_umat (SmartPtr<CLContext> context, SmartPtr<DrmBoBuffer> buffer, cv::UMat &image)
{
    SmartPtr<CLBuffer> cl_buffer = new CLVaBuffer (context, buffer);
    VideoBufferInfo info = buffer->get_video_info ();
    cl_mem cl_mem_id = cl_buffer->get_mem_id ();

    cv::ocl::convertFromBuffer (cl_mem_id, info.strides[0], info.height, info.width, CV_8U, image);
    if (image.empty ()) {
        XCAM_LOG_ERROR ("convert bo buffer to UMat failed");
        return false;
    }

    return true;
}

static void
add_detected_data (cv::UMat image, cv::Ptr<cv::Feature2D> detector, std::vector<cv::Point2f> &corners)
{
    std::vector<cv::KeyPoint> keypoints;
    detector->detect (image, keypoints);
    corners.reserve (corners.size () + keypoints.size ());
    for (size_t i = 0; i < keypoints.size (); ++i) {
        cv::KeyPoint &kp = keypoints[i];
        corners.push_back (kp.pt);
    }
}

static void
get_valid_offsets (
    cv::UMat out_image, cv::Size img0_size,
    std::vector<cv::Point2f> corner0, std::vector<cv::Point2f> corner1,
    std::vector<uchar> status, std::vector<float> error,
    std::vector<float> &offsets, float &sum, int &count)
{
    count = 0;
    sum = 0.0f;
    for (uint32_t i = 0; i < status.size (); ++i) {
#if XCAM_OF_DEBUG
        cv::Point start = cv::Point(corner0[i]) * XCAM_OF_DRAW_SCALE;
        cv::circle (out_image, start, 4, cv::Scalar(255, 255, 255), XCAM_OF_DRAW_SCALE);
#endif
        if (!status[i] || error[i] > 16)
            continue;
        if (fabs(corner0[i].y - corner1[i].y) >= 4)
            continue;

        float offset = corner1[i].x - corner0[i].x;

        sum += offset;
        ++count;
        offsets.push_back (offset);

#if XCAM_OF_DEBUG
        cv::Point end = (cv::Point(corner1[i]) + cv::Point (img0_size.width, 0)) * XCAM_OF_DRAW_SCALE;
        cv::line (out_image, start, end, cv::Scalar(255, 255, 255), XCAM_OF_DRAW_SCALE);
#else
        XCAM_UNUSED (out_image);
        XCAM_UNUSED (img0_size);
#endif
    }
}

static bool
get_mean_offset (std::vector<float> offsets, float sum, int &count, float &mean_offset)
{
    if (count < min_corners)
        return false;

    mean_offset = sum / count;

#if XCAM_OF_DEBUG
    XCAM_LOG_INFO (
        "X-axis mean offset:%.2f, pre_mean_offset:%.2f (%d times, count:%d)",
        mean_offset, 0.0f, 0, count);
#endif

    bool ret = true;
    float delta = 20.0f;//mean_offset;
    float pre_mean_offset = mean_offset;
    for (int try_times = 1; try_times < 4; ++try_times) {
        int recur_count = 0;
        sum = 0.0f;
        for (size_t i = 0; i < offsets.size (); ++i) {
            if (fabs (offsets[i] - mean_offset) >= 4.0f)
                continue;
            sum += offsets[i];
            ++recur_count;
        }

        if (recur_count < min_corners) {
            ret = false;
            break;
        }

        mean_offset = sum / recur_count;

#if XCAM_OF_DEBUG
        XCAM_LOG_INFO (
            "X-axis mean offset:%.2f, pre_mean_offset:%.2f (%d times, count:%d)",
            mean_offset, pre_mean_offset, try_times, recur_count);
#endif
        if (fabs (mean_offset - pre_mean_offset) > fabs (delta) * 1.2f) {
            ret = false;
            break;
        }

        delta = mean_offset - pre_mean_offset;
        pre_mean_offset = mean_offset;
        count = recur_count;
    }

    return ret;
}

static cv::UMat
calc_of_match (
    cv::UMat image0, cv::UMat image1,
    std::vector<cv::Point2f> corner0, std::vector<cv::Point2f> corner1,
    std::vector<uchar> &status, std::vector<float> &error,
    int &last_count, float &last_mean_offset, float &out_x_offset)
{
    cv::UMat out_image;
    cv::Size img0_size = image0.size ();
    cv::Size img1_size = image1.size ();
    XCAM_ASSERT (img0_size.height == img1_size.height);

#if XCAM_OF_DEBUG
    cv::Size size (img0_size.width + img1_size.width, img0_size.height);
    out_image.create (size, image0.type ());
    image0.copyTo (out_image (cv::Rect(0, 0, img0_size.width, img0_size.height)));
    image1.copyTo (out_image (cv::Rect(img0_size.width, 0, img1_size.width, img1_size.height)));

    cv::Size scale_size = size * XCAM_OF_DRAW_SCALE;
    cv::resize (out_image, out_image, scale_size, 0, 0);
#endif

    std::vector<float> offsets;
    float offset_sum = 0.0f;
    int count = 0;
    float mean_offset = 0.0f;
    offsets.reserve (corner0.size ());
    get_valid_offsets (out_image, img0_size, corner0, corner1, status, error,
                       offsets, offset_sum, count);

    bool ret = get_mean_offset (offsets, offset_sum, count, mean_offset);
    if (ret) {
        if (fabs (mean_offset - last_mean_offset) < delta_mean_offset ||
                fabs (mean_offset - out_x_offset) < delta_offset) {
            out_x_offset = out_x_offset * offset_factor + mean_offset * (1.0f - offset_factor);
        }
    } else
        out_x_offset = 0.0f;

    last_count = count;
    last_mean_offset = mean_offset;

    return out_image;
}

static void
adjust_stitch_area (int dst_width, cv::Rect &stitch0, cv::Rect &stitch1)
{
    int final_overlap_width = stitch1.x + stitch1.width + (dst_width - (stitch0.x + stitch0.width));
    final_overlap_width = XCAM_ALIGN_AROUND (final_overlap_width, 8);
    XCAM_ASSERT (final_overlap_width >= sitch_min_width);
    int center = final_overlap_width / 2;
    XCAM_ASSERT (center > sitch_min_width / 2);

    stitch1.x = XCAM_ALIGN_AROUND (center - sitch_min_width / 2, 8);
    stitch1.width = sitch_min_width;
    stitch0.x = dst_width - final_overlap_width + stitch1.x;
    stitch0.width = sitch_min_width;
}

static void
detect_and_match (
    cv::UMat img_left, cv::UMat img_right, cv::Rect &crop_left, cv::Rect &crop_right,
    int &valid_count, float &mean_offset, float &x_offset, int dst_width)
{
    std::vector<float> err;
    std::vector<uchar> status;
    std::vector<cv::Point2f> corner_left, corner_right;
    cv::Ptr<cv::Feature2D> fast_detector;

    fast_detector = cv::FastFeatureDetector::create (20, true);
    add_detected_data (img_left, fast_detector, corner_left);
    if (corner_left.empty ()) {
        valid_count = 0;
        mean_offset = 0.0f;
        x_offset = 0.0f;
        return;
    }

    cv::calcOpticalFlowPyrLK (
        img_left, img_right, corner_left, corner_right,
        status, err, cv::Size (16, 16), 3,
        cv::TermCriteria (cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 10, 0.01));
    cv::UMat img_out = calc_of_match (img_left, img_right, corner_left, corner_right,
                                      status, err, valid_count, mean_offset, x_offset);

    cv::Rect tmp_crop_left = crop_left;
    cv::Rect tmp_crop_right = crop_right;
    crop_left.x -= x_offset;
    adjust_stitch_area (dst_width, crop_left, crop_right);
    if (crop_left.x != tmp_crop_left.x || crop_left.width != tmp_crop_left.width ||
            crop_right.x != tmp_crop_right.x || crop_right.width != tmp_crop_right.width)
        x_offset = 0.0f;

#if XCAM_OF_DEBUG
    static int idx = 0;
    static int frame_num = 0;
    XCAM_LOG_INFO (
        "Stiching area %d: left_area(x:%d, width:%d), right_area(x:%d, width:%d)",
        idx, crop_left.x, crop_left.width, crop_right.x, crop_right.width);

    char file_name[1024];
    snprintf (file_name, 1024, "feature_match_%d_OF_%d.jpg", frame_num, idx);
    cv::imwrite (file_name, img_out);
    XCAM_LOG_INFO ("write feature match: %s", file_name);

    if (idx == 1)
        frame_num++;
    idx = (idx == 0) ? 1 : 0;
#endif
}

void
optical_flow_feature_match (
    SmartPtr<CLContext> context, int dst_width,
    SmartPtr<DrmBoBuffer> buf0, SmartPtr<DrmBoBuffer> buf1,
    cv::Rect &image0_crop_left, cv::Rect &image0_crop_right,
    cv::Rect &image1_crop_left, cv::Rect &image1_crop_right)
{
    cv::UMat image0, image1;
    cv::UMat image0_left, image0_right, image1_left, image1_right;
    static float x_offset0 = 0.0f, x_offset1 = 0.0f;
    static int valid_count0 = 0, valid_count1 = 0;
    static float mean_offset0 = 0.0f, mean_offset1 = 0.0f;

    if (!convert_to_umat (context, buf0, image0) || !convert_to_umat (context, buf1, image1))
        return;

    image0_left = image0 (image0_crop_left);
    image0_right = image0 (image0_crop_right);
    image1_left = image1 (image1_crop_left);
    image1_right = image1 (image1_crop_right);

    detect_and_match (image1_right, image0_left, image1_crop_right, image0_crop_left,
                      valid_count0, mean_offset0, x_offset0, dst_width);
    detect_and_match (image0_right, image1_left, image0_crop_right, image1_crop_left,
                      valid_count1, mean_offset1, x_offset1, dst_width);
}

