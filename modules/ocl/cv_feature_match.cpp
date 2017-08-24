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

namespace XCam {

#define XCAM_CV_FM_DEBUG 0
#define XCAM_CV_OF_DRAW_SCALE 2

#if XCAM_CV_FM_DEBUG
static XCamReturn
dump_buffer (SmartPtr<DrmBoBuffer> buffer, char *dump_name)
{
    ImageFileHandle file;

    XCamReturn ret = file.open (dump_name, "wb");
    if (ret != XCAM_RETURN_NO_ERROR) {
        XCAM_LOG_ERROR ("open %s failed", dump_name);
        return ret;
    }

    ret = file.write_buf (buffer);
    if (ret != XCAM_RETURN_NO_ERROR) {
        XCAM_LOG_ERROR ("write buffer to %s failed", dump_name);
        file.close ();
        return ret;
    }

    file.close ();

    return XCAM_RETURN_NO_ERROR;
}
#endif

CVFeatureMatch::CVFeatureMatch ()
    : CVBaseClass()
    , _x_offset (0.0f)
    , _mean_offset (0.0f)
    , _valid_count (0)
    , _fm_idx (-1)
    , _frame_num (0)
{
}

void
CVFeatureMatch::set_config (CVFMConfig config)
{
    _config = config;
}

CVFMConfig
CVFeatureMatch::get_config ()
{
    return _config;
}

void
CVFeatureMatch::set_fm_index (int idx)
{
    _fm_idx = idx;
}

bool
CVFeatureMatch::get_crop_image (
    SmartPtr<DrmBoBuffer> buffer, cv::Rect img_crop, cv::UMat &img)
{
    SmartPtr<CLBuffer> cl_buffer = new CLVaBuffer (_context, buffer);
    VideoBufferInfo info = buffer->get_video_info ();
    cl_mem cl_mem_id = cl_buffer->get_mem_id ();

    cv::UMat umat;
    cv::ocl::convertFromBuffer (cl_mem_id, info.strides[0], info.height, info.width, CV_8U, umat);
    if (umat.empty ()) {
        XCAM_LOG_ERROR ("convert bo buffer to UMat failed");
        return false;
    }

    img = umat (img_crop);

    return true;
}

void
CVFeatureMatch::add_detected_data (
    cv::InputArray image, cv::Ptr<cv::Feature2D> detector, std::vector<cv::Point2f> &corners)
{
    std::vector<cv::KeyPoint> keypoints;
    detector->detect (image, keypoints);
    corners.reserve (corners.size () + keypoints.size ());
    for (size_t i = 0; i < keypoints.size (); ++i) {
        cv::KeyPoint &kp = keypoints[i];
        corners.push_back (kp.pt);
    }
}

void
CVFeatureMatch::get_valid_offsets (
    cv::InputOutputArray image, cv::Size img0_size,
    std::vector<cv::Point2f> corner0, std::vector<cv::Point2f> corner1,
    std::vector<uchar> status, std::vector<float> error,
    std::vector<float> &offsets, float &sum, int &count)
{
    count = 0;
    sum = 0.0f;
    for (uint32_t i = 0; i < status.size (); ++i) {
#if XCAM_CV_FM_DEBUG
        cv::Point start = cv::Point(corner0[i]) * XCAM_CV_OF_DRAW_SCALE;
        cv::circle (image, start, 4, cv::Scalar(255, 255, 255), XCAM_CV_OF_DRAW_SCALE);
#endif

        if (!status[i] || error[i] > 24)
            continue;
        if (fabs(corner0[i].y - corner1[i].y) >= 8)
            continue;

        float offset = corner1[i].x - corner0[i].x;
        sum += offset;
        ++count;
        offsets.push_back (offset);

#if XCAM_CV_FM_DEBUG
        cv::Point end = (cv::Point(corner1[i]) + cv::Point (img0_size.width, 0)) * XCAM_CV_OF_DRAW_SCALE;
        cv::line (image, start, end, cv::Scalar(255, 255, 255), XCAM_CV_OF_DRAW_SCALE);
#else
        XCAM_UNUSED (image);
        XCAM_UNUSED (img0_size);
#endif
    }
}

bool
CVFeatureMatch::get_mean_offset (std::vector<float> offsets, float sum, int &count, float &mean_offset)
{
    if (count < _config.min_corners)
        return false;

    mean_offset = sum / count;

#if XCAM_CV_FM_DEBUG
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
            if (fabs (offsets[i] - mean_offset) >= 8.0f)
                continue;
            sum += offsets[i];
            ++recur_count;
        }

        if (recur_count < _config.min_corners) {
            ret = false;
            break;
        }

        mean_offset = sum / recur_count;
#if XCAM_CV_FM_DEBUG
        XCAM_LOG_INFO (
            "X-axis mean offset:%.2f, pre_mean_offset:%.2f (%d times, count:%d)",
            mean_offset, pre_mean_offset, try_times, recur_count);
#endif

        if (mean_offset == pre_mean_offset && recur_count == count)
            return true;

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

void
CVFeatureMatch::calc_of_match (
    cv::InputArray image0, cv::InputArray image1,
    std::vector<cv::Point2f> corner0, std::vector<cv::Point2f> corner1,
    std::vector<uchar> &status, std::vector<float> &error,
    int &last_count, float &last_mean_offset, float &out_x_offset)
{
    cv::_InputOutputArray out_image;
    cv::Size img0_size = image0.size ();
    cv::Size img1_size = image1.size ();
    XCAM_ASSERT (img0_size.height == img1_size.height);

#if XCAM_CV_FM_DEBUG
    cv::Mat mat;
    cv::UMat umat;
    cv::Size size (img0_size.width + img1_size.width, img0_size.height);

    if (image0.isUMat ()) {
        umat.create (size, image0.type ());
        out_image = cv::_InputOutputArray (umat);

        image0.copyTo (umat (cv::Rect(0, 0, img0_size.width, img0_size.height)));
        image1.copyTo (umat (cv::Rect(img0_size.width, 0, img1_size.width, img1_size.height)));
        umat.copyTo (out_image);
    } else {
        mat.create (size, image0.type ());
        out_image = cv::_InputOutputArray (mat);

        image0.copyTo (mat (cv::Rect(0, 0, img0_size.width, img0_size.height)));
        image1.copyTo (mat (cv::Rect(img0_size.width, 0, img1_size.width, img1_size.height)));
        mat.copyTo (out_image);
    }

    cv::Size scale_size = size * XCAM_CV_OF_DRAW_SCALE;
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
        if (fabs (mean_offset - last_mean_offset) < _config.delta_mean_offset) {
            out_x_offset = out_x_offset * _config.offset_factor + mean_offset * (1.0f - _config.offset_factor);

            if (fabs (out_x_offset) > _config.max_adjusted_offset)
                out_x_offset = (out_x_offset > 0.0f) ? _config.max_adjusted_offset : (-_config.max_adjusted_offset);
        }
    }

    last_count = count;
    last_mean_offset = mean_offset;

#if XCAM_CV_FM_DEBUG
    char file_name[1024];
    snprintf (file_name, 1023, "fm_optical_flow_%d_%d.jpg", _frame_num, _fm_idx);
    cv::imwrite (file_name, out_image);
#endif
}

void
CVFeatureMatch::adjust_stitch_area (int dst_width, float &x_offset, cv::Rect &stitch0, cv::Rect &stitch1)
{
    if (fabs (x_offset) < 5.0f)
        return;

    int last_overlap_width = stitch1.x + stitch1.width + (dst_width - (stitch0.x + stitch0.width));
    // int final_overlap_width = stitch1.x + stitch1.width + (dst_width - (stitch0.x - x_offset + stitch0.width));
    int final_overlap_width = last_overlap_width + x_offset;
    final_overlap_width = XCAM_ALIGN_AROUND (final_overlap_width, 8);
    XCAM_ASSERT (final_overlap_width >= _config.sitch_min_width);
    int center = final_overlap_width / 2;
    XCAM_ASSERT (center > _config.sitch_min_width / 2);

    stitch1.x = XCAM_ALIGN_AROUND (center - _config.sitch_min_width / 2, 8);
    stitch1.width = _config.sitch_min_width;
    stitch0.x = dst_width - final_overlap_width + stitch1.x;
    stitch0.width = _config.sitch_min_width;

    float delta_offset = final_overlap_width - last_overlap_width;
    x_offset -= delta_offset;
}

void
CVFeatureMatch::detect_and_match (
    cv::InputArray img_left, cv::InputArray img_right, cv::Rect &crop_left, cv::Rect &crop_right,
    int &valid_count, float &mean_offset, float &x_offset, int dst_width)
{
    std::vector<float> err;
    std::vector<uchar> status;
    std::vector<cv::Point2f> corner_left, corner_right;
    cv::Ptr<cv::Feature2D> fast_detector;
    cv::Size win_size = cv::Size (5, 5);

    if (img_left.isUMat ())
        win_size = cv::Size (16, 16);

    fast_detector = cv::FastFeatureDetector::create (20, true);
    add_detected_data (img_left, fast_detector, corner_left);

    if (corner_left.empty ()) {
        return;
    }

    cv::calcOpticalFlowPyrLK (
        img_left, img_right, corner_left, corner_right, status, err, win_size, 3,
        cv::TermCriteria (cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 10, 0.01f));
    cv::ocl::finish();

    calc_of_match (img_left, img_right, corner_left, corner_right,
                   status, err, valid_count, mean_offset, x_offset);
    adjust_stitch_area (dst_width, x_offset, crop_left, crop_right);

#if XCAM_CV_FM_DEBUG
    XCAM_LOG_INFO (
        "Stiching area: left_area(x:%d, width:%d), right_area(x:%d, width:%d)",
        crop_left.x, crop_left.width, crop_right.x, crop_right.width);
#endif
}

void
CVFeatureMatch::optical_flow_feature_match (
    SmartPtr<DrmBoBuffer> left_buf, SmartPtr<DrmBoBuffer> right_buf,
    cv::Rect &left_img_crop, cv::Rect &right_img_crop, int dst_width)
{
    cv::UMat left_umat, right_umat;
    cv::Mat left_mat, right_mat;
    cv::_InputArray left_img, right_img;

    if (!get_crop_image (left_buf, left_img_crop, left_umat)
            || !get_crop_image (right_buf, right_img_crop, right_umat))
        return;

    if (_use_ocl) {
        left_img = cv::_InputArray (left_umat);
        right_img = cv::_InputArray (right_umat);
    } else {
        left_mat = left_umat.getMat (cv::ACCESS_READ);
        right_mat = right_umat.getMat (cv::ACCESS_READ);

        left_img = cv::_InputArray (left_mat);
        right_img = cv::_InputArray (right_mat);
    }

    detect_and_match (left_img, right_img, left_img_crop, right_img_crop,
                      _valid_count, _mean_offset, _x_offset, dst_width);

#if XCAM_CV_FM_DEBUG
    XCAM_ASSERT (_fm_idx >= 0);
    char file_name[1024];

    VideoBufferInfo info = left_buf->get_video_info ();
    std::snprintf (file_name, 1023, "fm_in_%d_%d_%dx%d_0.nv12", info.width, info.height, _frame_num, _fm_idx);
    dump_buffer (left_buf, file_name);

    info = right_buf->get_video_info ();
    std::snprintf (file_name, 1023, "fm_in_%d_%d_%dx%d_1.nv12", info.width, info.height, _frame_num, _fm_idx);
    dump_buffer (right_buf, file_name);

    cv::Mat mat;
    std::snprintf (file_name, 1023, "fm_in_stitch_area_%d_%d_0.jpg", _frame_num, _fm_idx);
    convert_to_mat (left_buf, mat);
    cv::line (mat, cv::Point(left_img_crop.x, 0), cv::Point(left_img_crop.x, dst_width), cv::Scalar(0, 0, 255), 2);
    cv::line (mat, cv::Point(left_img_crop.x + left_img_crop.width, 0),
              cv::Point(left_img_crop.x + left_img_crop.width, dst_width), cv::Scalar(0, 0, 255), 2);
    cv::imwrite (file_name, mat);

    std::snprintf (file_name, 1023, "fm_in_stitch_area_%d_%d_1.jpg", _frame_num, _fm_idx);
    convert_to_mat (right_buf, mat);
    cv::line (mat, cv::Point(right_img_crop.x, 0), cv::Point(right_img_crop.x, dst_width), cv::Scalar(0, 0, 255), 2);
    cv::line (mat, cv::Point(right_img_crop.x + right_img_crop.width, 0),
              cv::Point(right_img_crop.x + right_img_crop.width, dst_width), cv::Scalar(0, 0, 255), 2);
    cv::imwrite (file_name, mat);

    XCAM_LOG_INFO ("Feature match: frame number:%d index:%d done", _frame_num, _fm_idx);
    _frame_num++;
#endif
}

}
