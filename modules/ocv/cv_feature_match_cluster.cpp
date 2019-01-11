/*
 * cv_feature_match_cluster.cpp - optical flow feature match selected by clustering
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
 * Author: Wu Junkai <junkai.wu@intel.com>
 */

#include "xcam_obj_debug.h"
#include "image_file_handle.h"
#include "cv_feature_match_cluster.h"

#define XCAM_CV_FM_DEBUG 0
#define XCAM_CV_OF_DRAW_SCALE 2

namespace XCam {
CVFeatureMatchCluster::CVFeatureMatchCluster ()
    : CVFeatureMatch ()
{
}

bool
CVFeatureMatchCluster::calc_mean_offset (
    std::vector<cv::Point2f> &corner0, std::vector<cv::Point2f> &corner1,
    std::vector<uchar> &status, std::vector<float> &error,
    float &mean_offset_x, float &mean_offset_y,
    cv::Mat debug_img, cv::Size &img0_size, cv::Size &img1_size)
{
    std::vector<std::vector<uint32_t>> clusters;
    std::vector<std::vector<cv::Point2f>> clusters_offsets;
    std::vector<uint32_t> valid_seeds (status.size ());
    std::vector<uint32_t> valid_corners (status.size ());

    for (uint32_t i = 0; i < status.size (); ++i) {
        if (!status[i] || (error[i] > _config.max_track_error) || corner1[i].x < 0.0f || corner1[i].x > img0_size.width) {
            valid_corners[i] = 0;
            valid_seeds[i] = 0;
        } else {
            valid_corners[i] = 1;
            valid_seeds[i] = 1;
        }
    }

    float seed_x_offset = 0.0f;
    float seed_y_offset = 0.0f;

    std::vector<uint32_t> cluster (1);
    std::vector<cv::Point2f> cluster_offset (1);

    float thres = 8.0f;
    while (cluster.size() > 0) {
        cluster.clear ();
        cluster_offset.clear ();

        for (uint32_t i = 0; i < status.size (); ++i) {
            if (valid_seeds[i]) {
                seed_x_offset = corner1[i].x - corner0[i].x;
                seed_y_offset = corner1[i].y - corner0[i].y;
                cluster.push_back (i);
                cluster_offset.push_back (cv::Point2f(seed_x_offset, seed_y_offset));
                valid_corners[i] = 0;
                valid_seeds[i] = 0;
                break;
            }
        }

        if (cluster.size() > 0) {
            for (uint32_t i = 0; i < status.size (); ++i) {
                if (!valid_corners[i])
                    continue;

                float x_offset = corner1[i].x - corner0[i].x;
                float y_offset = corner1[i].y - corner0[i].y;

                if (fabs (x_offset - seed_x_offset) > thres || fabs (y_offset - seed_y_offset) > thres / 2.0f)
                    continue;

                cluster.push_back (i);
                cluster_offset.push_back (cv::Point2f(x_offset, y_offset));
                valid_seeds[i] = 0;
            }

            clusters.push_back (cluster);
            clusters_offsets.push_back (cluster_offset);
        }
    }

    if (clusters_offsets.size () == 0)
        return false;

    uint32_t max_size = 0;
    uint32_t max_index = 0;

    for (uint32_t i = 0; i < clusters.size (); ++i) {
        if (clusters[i].size () > max_size) {
            max_size = clusters[i].size ();
            max_index = i;
        }
    }

    if (clusters_offsets[max_index].size () < (uint32_t)_config.min_corners)
        return false;

    float sum_x = 0.0f;
    float sum_y = 0.0f;
    for (uint32_t i = 0; i < clusters_offsets[max_index].size (); ++i) {
        sum_x += clusters_offsets[max_index][i].x;
        sum_y += clusters_offsets[max_index][i].y;
    }

    mean_offset_x = sum_x / clusters_offsets[max_index].size ();
    mean_offset_y = sum_y / clusters_offsets[max_index].size ();

#if XCAM_CV_FM_DEBUG
    for (uint32_t i = 0; i < status.size (); ++i) {
        if(!status[i])
            continue;

        cv::Point start = cv::Point(corner0[i]) * XCAM_CV_OF_DRAW_SCALE;
        cv::circle (debug_img, start, 4, cv::Scalar(0), XCAM_CV_OF_DRAW_SCALE);
        cv::Point end = (cv::Point(corner1[i]) + cv::Point (img0_size.width, 0)) * XCAM_CV_OF_DRAW_SCALE;
        cv::line (debug_img, start, end, cv::Scalar(0), XCAM_CV_OF_DRAW_SCALE);
    }

    for (uint32_t i = 0; i < status.size (); ++i) {
        if (!status[i])
            continue;

        cv::Point start = cv::Point(corner0[i]) * XCAM_CV_OF_DRAW_SCALE;
        cv::circle (debug_img, start, 4, cv::Scalar(0), XCAM_CV_OF_DRAW_SCALE);
        if (error[i] > _config.max_track_error)
            continue;
        if (fabs(corner0[i].y - corner1[i].y) >= _config.max_valid_offset_y)
            continue;
        if (corner1[i].x < 0.0f || corner1[i].x > img0_size.width)
            continue;

        cv::Point end = (cv::Point(corner1[i]) + cv::Point (img0_size.width, 0)) * XCAM_CV_OF_DRAW_SCALE;
        cv::line (debug_img, start, end, cv::Scalar(255), XCAM_CV_OF_DRAW_SCALE);
    }

    for (uint32_t i = 0; i < status.size (); ++i) {
        if(!status[i])
            continue;

        cv::Point start = (cv::Point(corner0[i]) + cv::Point (img0_size.width + img1_size.width, 0)) * XCAM_CV_OF_DRAW_SCALE;
        cv::circle (debug_img, start, 4, cv::Scalar(0), XCAM_CV_OF_DRAW_SCALE);
        cv::Point end = (cv::Point(corner1[i]) + cv::Point (2 * img0_size.width + img1_size.width, 0)) * XCAM_CV_OF_DRAW_SCALE;
        cv::line (debug_img, start, end, cv::Scalar(0), XCAM_CV_OF_DRAW_SCALE);
    }

    if (clusters.size () != 0)
        cluster = clusters[max_index];
    for (uint32_t i = 0; i < cluster.size (); ++i) {
        cv::Point start = (cv::Point(corner0[cluster[i]]) + cv::Point(img0_size.width + img1_size.width, 0)) * XCAM_CV_OF_DRAW_SCALE;
        cv::circle (debug_img, start, 4, cv::Scalar(0), XCAM_CV_OF_DRAW_SCALE);
        cv::Point end = (cv::Point(corner1[cluster[i]]) + cv::Point (2 * img0_size.width + img1_size.width, 0)) * XCAM_CV_OF_DRAW_SCALE;
        cv::line (debug_img, start, end, cv::Scalar(255), XCAM_CV_OF_DRAW_SCALE);
    }

#endif

    XCAM_UNUSED (debug_img);
    XCAM_UNUSED (img0_size);
    XCAM_UNUSED (img1_size);

    clusters.clear ();
    clusters_offsets.clear ();

    return true;
}

void
CVFeatureMatchCluster::calc_of_match (
    cv::Mat image0, cv::Mat image1, std::vector<cv::Point2f> &corner0, std::vector<cv::Point2f> &corner1,
    std::vector<uchar> &status, std::vector<float> &error)
{
    cv::Mat debug_img;
    cv::Size img0_size = image0.size ();
    cv::Size img1_size = image1.size ();
    XCAM_ASSERT (img0_size.height == img1_size.height);

#if XCAM_CV_FM_DEBUG
    cv::Mat mat;
    cv::Size size ((img0_size.width + img1_size.width) * 2, img0_size.height);

    mat.create (size, image0.type ());
    debug_img = cv::Mat (mat);

    image0.copyTo (mat (cv::Rect(0, 0, img0_size.width, img0_size.height)));
    image1.copyTo (mat (cv::Rect(img0_size.width, 0, img1_size.width, img1_size.height)));
    image0.copyTo (mat (cv::Rect(img0_size.width + img1_size.width, 0, img0_size.width, img0_size.height)));
    image1.copyTo (mat (cv::Rect(2 * img0_size.width + img1_size.width, 0, img1_size.width, img1_size.height)));

    mat.copyTo (debug_img);

    cv::Size scale_size = size * XCAM_CV_OF_DRAW_SCALE;
    cv::resize (debug_img, debug_img, scale_size, 0, 0);
#endif

    float mean_offset_x = 0.0f;
    float mean_offset_y = 0.0f;
    float last_mean_offset_x = _mean_offset;
    float last_mean_offset_y = _mean_offset_y;
    bool ret = calc_mean_offset (corner0, corner1, status, error, mean_offset_x, mean_offset_y,
                                 debug_img, img0_size, img1_size);

#if XCAM_CV_FM_DEBUG
    char file_name[256];
    std::snprintf (file_name, 256, "fm_optical_flow_%d_%d.jpg", _frame_num, _fm_idx);
    cv::imwrite (file_name, debug_img);
#endif

    if (ret) {
        if (fabs (mean_offset_x - last_mean_offset_x) < _config.delta_mean_offset) {
            _x_offset = _x_offset * _config.offset_factor + mean_offset_x * (1.0f - _config.offset_factor);

            if (fabs (_x_offset) > _config.max_adjusted_offset)
                _x_offset = (_x_offset > 0.0f) ? _config.max_adjusted_offset : (-_config.max_adjusted_offset);
        }

        if (fabs (mean_offset_y - last_mean_offset_y) < _config.delta_mean_offset) {
            _y_offset = _y_offset * _config.offset_factor + mean_offset_y * (1.0f - _config.offset_factor);

            if (fabs (_y_offset) > _config.max_adjusted_offset)
                _y_offset = (_y_offset > 0.0f) ? _config.max_adjusted_offset : (-_config.max_adjusted_offset);
        }
    }

    _mean_offset = mean_offset_x;
    _mean_offset_y = mean_offset_y;
}

void
CVFeatureMatchCluster::detect_and_match (cv::Mat img_left, cv::Mat img_right)
{
    std::vector<float> err;
    std::vector<uchar> status;
    std::vector<cv::Point2f> corner_left, corner_right;
    cv::Ptr<cv::Feature2D> fast_detector;
    cv::Size win_size = cv::Size (21, 21);

    fast_detector = cv::FastFeatureDetector::create (20, true);
    add_detected_data (img_left, fast_detector, corner_left);

    if (corner_left.empty ()) {
        return;
    }

    cv::calcOpticalFlowPyrLK (
        img_left, img_right, corner_left, corner_right, status, err, win_size, 3,
        cv::TermCriteria (cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01f));

    calc_of_match (img_left, img_right, corner_left, corner_right, status, err);

#if XCAM_CV_FM_DEBUG
    XCAM_LOG_INFO ("x_offset:%0.2f", _x_offset);
    XCAM_LOG_INFO (
        "FeatureMatch(idx:%d): stiching area: left_area(pos_x:%d, width:%d), right_area(pos_x:%d, width:%d)",
        _fm_idx, _left_rect.pos_x, _left_rect.width, _right_rect.pos_x, _right_rect.width);
#endif
}

void
CVFeatureMatchCluster::feature_match (
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

SmartPtr<FeatureMatch>
FeatureMatch::create_cluster_feature_match ()
{
    SmartPtr<CVFeatureMatchCluster> matcher = new CVFeatureMatchCluster ();
    XCAM_ASSERT (matcher.ptr ());

    return matcher;
}

}
