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

#include "xcam_obj_debug.h"
#include "image_file_handle.h"
#include "cv_feature_match.h"
#if HAVE_LIBCL
#include <opencv2/core/ocl.hpp>
#endif

#define XCAM_CV_FM_DEBUG 0
#define XCAM_CV_OF_DRAW_SCALE 2

namespace XCam {
CVFeatureMatch::CVFeatureMatch ()
    : FeatureMatch ()
    , _dst_width (0)
    , _need_adjust (false)
{
    xcam_mem_clear (_cl_buf_mem);
}

CVFeatureMatch::~CVFeatureMatch ()
{
    xcam_mem_clear (_cl_buf_mem);
}


void
CVFeatureMatch::set_dst_width (int width)
{
    _dst_width = width;
}

void
CVFeatureMatch::enable_adjust_crop_area ()
{
    _need_adjust = true;
}

void
CVFeatureMatch::set_cl_buf_mem (void *mem, BufId id)
{
#if HAVE_LIBCL
    XCAM_ASSERT (mem);
    _cl_buf_mem[id] = mem;
#else
    XCAM_LOG_DEBUG ("non-OpenCL mode, failed to set cl buffer memory");
#endif
}

bool
CVFeatureMatch::get_crop_image_umat (
    const SmartPtr<VideoBuffer> &buffer, const Rect &crop_rect, cv::UMat &img, BufId id)
{
#if HAVE_LIBCL
    VideoBufferInfo info = buffer->get_video_info ();

    cv::UMat umat;
    cv::ocl::convertFromBuffer (_cl_buf_mem[id], info.strides[0], info.height, info.width, CV_8U, umat);
    if (umat.empty ()) {
        XCAM_LOG_ERROR ("FeatureMatch(idx:%d): convert bo buffer to UMat failed", _fm_idx);
        return false;
    }

    img = umat (cv::Rect(crop_rect.pos_x, crop_rect.pos_y, crop_rect.width, crop_rect.height));

    return true;
#else
    XCAM_LOG_ERROR ("FeatureMatch(idx:%d): non-OpenCL mode, failed to get umat", _fm_idx);
    return false;
#endif
}

void
CVFeatureMatch::add_detected_data (
    cv::Mat image, cv::Ptr<cv::Feature2D> detector, std::vector<cv::Point2f> &corners)
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
    std::vector<cv::Point2f> &corner0, std::vector<cv::Point2f> &corner1,
    std::vector<uchar> &status, std::vector<float> &error,
    std::vector<float> &offsets, float &sum, int &count,
    cv::Mat debug_img, cv::Size &img0_size)
{
    count = 0;
    sum = 0.0f;
    for (uint32_t i = 0; i < status.size (); ++i) {
        if (!status[i])
            continue;

#if XCAM_CV_FM_DEBUG
        cv::Point start = cv::Point(corner0[i]) * XCAM_CV_OF_DRAW_SCALE;
        cv::circle (debug_img, start, 4, cv::Scalar(255), XCAM_CV_OF_DRAW_SCALE);
#endif
        if (error[i] > _config.max_track_error)
            continue;
        if (fabs(corner0[i].y - corner1[i].y) >= _config.max_valid_offset_y)
            continue;
        if (corner1[i].x < 0.0f || corner1[i].x > img0_size.width)
            continue;

        float offset = corner1[i].x - corner0[i].x;
        sum += offset;
        ++count;
        offsets.push_back (offset);

#if XCAM_CV_FM_DEBUG
        cv::Point end = (cv::Point(corner1[i]) + cv::Point (img0_size.width, 0)) * XCAM_CV_OF_DRAW_SCALE;
        cv::line (debug_img, start, end, cv::Scalar(255), XCAM_CV_OF_DRAW_SCALE);
#else
        XCAM_UNUSED (debug_img);
        XCAM_UNUSED (img0_size);
#endif
    }
}

void
CVFeatureMatch::calc_of_match (
    cv::Mat image0, cv::Mat image1, std::vector<cv::Point2f> &corner0, std::vector<cv::Point2f> &corner1,
    std::vector<uchar> &status, std::vector<float> &error)
{
    cv::Mat debug_img;
    cv::Size img0_size = image0.size ();
    cv::Size img1_size = image1.size ();
    XCAM_ASSERT (img0_size.height == img1_size.height);

#if XCAM_CV_FM_DEBUG
    cv::Mat mat;
    cv::Size size (img0_size.width + img1_size.width, img0_size.height);

    mat.create (size, image0.type ());
    debug_img = cv::Mat (mat);

    image0.copyTo (mat (cv::Rect(0, 0, img0_size.width, img0_size.height)));
    image1.copyTo (mat (cv::Rect(img0_size.width, 0, img1_size.width, img1_size.height)));
    mat.copyTo (debug_img);

    cv::Size scale_size = size * XCAM_CV_OF_DRAW_SCALE;
    cv::resize (debug_img, debug_img, scale_size, 0, 0);
#endif

    std::vector<float> offsets;
    float offset_sum = 0.0f;
    int count = 0;
    float mean_offset = 0.0f;
    float last_mean_offset = _mean_offset;
    offsets.reserve (corner0.size ());
    get_valid_offsets (corner0, corner1, status, error,
                       offsets, offset_sum, count, debug_img, img0_size);
#if XCAM_CV_FM_DEBUG
    XCAM_LOG_INFO ("FeatureMatch(idx:%d): valid offsets:%d", _fm_idx, offsets.size ());
    char file_name[256];
    std::snprintf (file_name, 256, "fm_optical_flow_%d_%d.jpg", _frame_num, _fm_idx);
    cv::imwrite (file_name, debug_img);
#endif

    bool ret = get_mean_offset (offsets, offset_sum, count, mean_offset);
    if (ret) {
        if (fabs (mean_offset - last_mean_offset) < _config.delta_mean_offset) {
            _x_offset = _x_offset * _config.offset_factor + mean_offset * (1.0f - _config.offset_factor);

            if (fabs (_x_offset) > _config.max_adjusted_offset)
                _x_offset = (_x_offset > 0.0f) ? _config.max_adjusted_offset : (-_config.max_adjusted_offset);
        }
    }

    _valid_count = count;
    _mean_offset = mean_offset;
}

void
CVFeatureMatch::adjust_crop_area ()
{
    if (fabs (_x_offset) < 5.0f)
        return;

    XCAM_ASSERT (_dst_width);

    int last_overlap_width = _right_rect.pos_x + _right_rect.width +
                              (_dst_width - (_left_rect.pos_x + _left_rect.width));
    // int final_overlap_width = _right_rect.pos_x + _right_rect.width +
    //                           (dst_width - (_left_rect.pos_x - x_offset + _left_rect.width));
    if ((_left_rect.pos_x - _x_offset + _left_rect.width) > _dst_width)
        _x_offset = _dst_width - (_left_rect.pos_x + _left_rect.width);
    int final_overlap_width = last_overlap_width + _x_offset;
    final_overlap_width = XCAM_ALIGN_AROUND (final_overlap_width, 8);
    XCAM_ASSERT (final_overlap_width >= _config.sitch_min_width);
    int center = final_overlap_width / 2;
    XCAM_ASSERT (center >= _config.sitch_min_width / 2);

    _right_rect.pos_x = XCAM_ALIGN_AROUND (center - _config.sitch_min_width / 2, 8);
    _right_rect.width = _config.sitch_min_width;
    _left_rect.pos_x = _dst_width - final_overlap_width + _right_rect.pos_x;
    _left_rect.width = _config.sitch_min_width;

    float delta_offset = final_overlap_width - last_overlap_width;
    _x_offset -= delta_offset;
}

void
CVFeatureMatch::detect_and_match (cv::Mat img_left, cv::Mat img_right)
{
    std::vector<float> err;
    std::vector<uchar> status;
    std::vector<cv::Point2f> corner_left, corner_right;
    cv::Ptr<cv::Feature2D> fast_detector;
    cv::Size win_size = cv::Size (5, 5);

    fast_detector = cv::FastFeatureDetector::create (20, true);
    add_detected_data (img_left, fast_detector, corner_left);

    if (corner_left.empty ()) {
        return;
    }

    cv::calcOpticalFlowPyrLK (
        img_left, img_right, corner_left, corner_right, status, err, win_size, 3,
        cv::TermCriteria (cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 10, 0.01f));

    calc_of_match (img_left, img_right, corner_left, corner_right, status, err);

    if (_need_adjust)
        adjust_crop_area ();

#if XCAM_CV_CAPI_FM_DEBUG
    XCAM_LOG_INFO ("FeatureMatch(idx:%d): x_offset:%0.2f", _fm_idx, _x_offset);
    if (_need_adjust) {
        XCAM_LOG_INFO (
            "FeatureMatch(idx:%d): stiching area: left_area(pos_x:%d, width:%d), right_area(pos_x:%d, width:%d)",
            _fm_idx, _left_rect.pos_x, _left_rect.width, _right_rect.pos_x, _right_rect.width);
    }
#endif
}

void
CVFeatureMatch::feature_match (
    const SmartPtr<VideoBuffer> &left_buf, const SmartPtr<VideoBuffer> &right_buf)
{
    XCAM_ASSERT (_left_rect.width && _left_rect.height);
    XCAM_ASSERT (_right_rect.width && _right_rect.height);

    cv::UMat left_umat, right_umat;
    cv::Mat left_img, right_img;

    if (_cl_buf_mem[BufIdLeft] && _cl_buf_mem[BufIdRight]) {
        if (!get_crop_image_umat (left_buf, _left_rect, left_umat, BufIdLeft)
                || !get_crop_image_umat (right_buf, _right_rect, right_umat, BufIdRight))
            return;

        left_img = left_umat.getMat (cv::ACCESS_READ);
        right_img = right_umat.getMat (cv::ACCESS_READ);
    } else {
        if (!convert_range_to_mat (left_buf, _left_rect, left_img)
                || !convert_range_to_mat (right_buf, _right_rect, right_img))
            return;
    }

    detect_and_match (left_img, right_img);

#if XCAM_CV_FM_DEBUG
    debug_write_image (left_buf, right_buf, _left_rect, _right_rect, _frame_num, _fm_idx);
    _frame_num++;
#endif
}

void
CVFeatureMatch::debug_write_image (
    const SmartPtr<VideoBuffer> &left_buf, const SmartPtr<VideoBuffer> &right_buf,
    const Rect &left_rect, const Rect &right_rect, uint32_t frame_num, int fm_idx)
{
    XCAM_ASSERT (fm_idx >= 0);

    char frame_str[64] = {'\0'};
    std::snprintf (frame_str, 64, "frame:%d", frame_num);
    char fm_idx_str[64] = {'\0'};
    std::snprintf (fm_idx_str, 64, "fm_idx:%d", fm_idx);

    char img_name[256] = {'\0'};
    std::snprintf (img_name, 256, "fm_in_stitch_area_%d_%d_0.jpg", frame_num, fm_idx);
    write_image (left_buf, left_rect, img_name, frame_str, fm_idx_str);

    std::snprintf (img_name, 256, "fm_in_stitch_area_%d_%d_1.jpg", frame_num, fm_idx);
    write_image (right_buf, right_rect, img_name, frame_str, fm_idx_str);

    XCAM_LOG_INFO ("FeatureMatch(idx:%d): frame number:%d done", fm_idx, frame_num);
}

SmartPtr<FeatureMatch>
FeatureMatch::create_default_feature_match ()
{
    SmartPtr<CVFeatureMatch> matcher = new CVFeatureMatch ();
    XCAM_ASSERT (matcher.ptr ());

    return matcher;
}

}
