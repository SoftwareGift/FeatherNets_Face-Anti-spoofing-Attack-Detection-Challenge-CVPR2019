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

#include "cv_feature_match_cluster.h"
#include "xcam_obj_debug.h"
#include "image_file_handle.h"
#include "cl_utils.h"

#define XCAM_CV_FM_DEBUG 0
#define XCAM_CV_OF_DRAW_SCALE 2

namespace XCam {
CVFeatureMatchCluster::CVFeatureMatchCluster ()
    : CVFeatureMatch ()
{
    XCAM_ASSERT (_cv_context.ptr ());
}

bool
CVFeatureMatchCluster::calc_mean_offset (
    std::vector<cv::Point2f> &corner0, std::vector<cv::Point2f> &corner1,
    std::vector<uchar> &status, std::vector<float> &error,
    float &mean_offset_x, float &mean_offset_y,
    cv::InputOutputArray debug_img, cv::Size &img0_size, cv::Size &img1_size)
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

                if (fabs (x_offset - seed_x_offset) > thres || fabs (y_offset - seed_y_offset) > thres)
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
CVFeatureMatchCluster::calc_of_match_cluster (
    cv::InputArray image0, cv::InputArray image1,
    std::vector<cv::Point2f> &corner0, std::vector<cv::Point2f> &corner1,
    std::vector<uchar> &status, std::vector<float> &error,
    float &last_mean_offset_x, float &last_mean_offset_y,
    float &out_x_offset, float &out_y_offset)
{
    cv::_InputOutputArray debug_img;
    cv::Size img0_size = image0.size ();
    cv::Size img1_size = image1.size ();
    XCAM_ASSERT (img0_size.height == image1.rows ());
    XCAM_UNUSED (image1);

#if XCAM_CV_FM_DEBUG
    cv::Mat mat;
    cv::UMat umat;
    cv::Size size ((img0_size.width + img1_size.width) * 2, img0_size.height);

    if (image0.isUMat ()) {
        umat.create (size, image0.type ());
        debug_img = cv::_InputOutputArray (umat);

        image0.copyTo (umat (cv::Rect(0, 0, img0_size.width, img0_size.height)));
        image1.copyTo (umat (cv::Rect(img0_size.width, 0, img1_size.width, img1_size.height)));

        image0.copyTo (umat (cv::Rect(img0_size.width + img1_size.width, 0, img0_size.width, img0_size.height)));
        image1.copyTo (umat (cv::Rect(2 * img0_size.width + img1_size.width, 0, img1_size.width, img1_size.height)));
        umat.copyTo (debug_img);
    } else {
        mat.create (size, image0.type ());
        debug_img = cv::_InputOutputArray (mat);

        image0.copyTo (mat (cv::Rect(0, 0, img0_size.width, img0_size.height)));
        image1.copyTo (mat (cv::Rect(img0_size.width, 0, img1_size.width, img1_size.height)));

        image0.copyTo (mat (cv::Rect(img0_size.width + img1_size.width, 0, img0_size.width, img0_size.height)));
        image1.copyTo (mat (cv::Rect(2 * img0_size.width + img1_size.width, 0, img1_size.width, img1_size.height)));

        mat.copyTo (debug_img);
    }

    cv::Size scale_size = size * XCAM_CV_OF_DRAW_SCALE;
    cv::resize (debug_img, debug_img, scale_size, 0, 0);
#endif

    float mean_offset_x = 0.0f;
    float mean_offset_y = 0.0f;
    bool ret = calc_mean_offset (corner0, corner1, status, error, mean_offset_x, mean_offset_y, debug_img, img0_size, img1_size);

#if XCAM_CV_FM_DEBUG
    char file_name[256];
    std::snprintf (file_name, 256, "fm_optical_flow_%d_%d.jpg", _frame_num, _fm_idx);
    cv::imwrite (file_name, debug_img);
#endif

    if (ret) {
        if (fabs (mean_offset_x - last_mean_offset_x) < _config.delta_mean_offset) {
            out_x_offset = out_x_offset * _config.offset_factor + mean_offset_x * (1.0f - _config.offset_factor);

            if (fabs (out_x_offset) > _config.max_adjusted_offset)
                out_x_offset = (out_x_offset > 0.0f) ? _config.max_adjusted_offset : (-_config.max_adjusted_offset);
        }

        if (fabs (mean_offset_y - last_mean_offset_y) < _config.delta_mean_offset) {
            out_y_offset = out_y_offset * _config.offset_factor + mean_offset_y * (1.0f - _config.offset_factor);

            if (fabs (out_y_offset) > _config.max_adjusted_offset)
                out_y_offset = (out_y_offset > 0.0f) ? _config.max_adjusted_offset : (-_config.max_adjusted_offset);
        }
    }

    last_mean_offset_x = mean_offset_x;
    last_mean_offset_y = mean_offset_y;
}

void
CVFeatureMatchCluster::detect_and_match_cluster (
    cv::InputArray img_left, cv::InputArray img_right, Rect &crop_left, Rect &crop_right,
    float &mean_offset_x, float &mean_offset_y, float &x_offset, float &y_offset)
{
    std::vector<float> err;
    std::vector<uchar> status;
    std::vector<cv::Point2f> corner_left, corner_right;
    cv::Ptr<cv::Feature2D> fast_detector;
    cv::Size win_size = cv::Size (21, 21);

    if (img_left.isUMat ())
        win_size = cv::Size (16, 16);

    fast_detector = cv::FastFeatureDetector::create (20, true);
    add_detected_data (img_left, fast_detector, corner_left);

    if (corner_left.empty ()) {
        return;
    }

    cv::calcOpticalFlowPyrLK (
        img_left, img_right, corner_left, corner_right, status, err, win_size, 3,
        cv::TermCriteria (cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01f));
    cv::ocl::finish();

    calc_of_match_cluster (img_left, img_right, corner_left, corner_right,
                           status, err, mean_offset_x, mean_offset_y, x_offset, y_offset);

#if XCAM_CV_FM_DEBUG
    XCAM_LOG_INFO ("x_offset:%0.2f", x_offset);
    XCAM_LOG_INFO (
        "FeatureMatch(idx:%d): stiching area: left_area(pos_x:%d, width:%d), right_area(pos_x:%d, width:%d)",
        _fm_idx, crop_left.pos_x, crop_left.width, crop_right.pos_x, crop_right.width);
#endif

    XCAM_UNUSED (crop_left);
    XCAM_UNUSED (crop_right);
}

void
CVFeatureMatchCluster::optical_flow_feature_match (
    const SmartPtr<VideoBuffer> &left_buf, const SmartPtr<VideoBuffer> &right_buf,
    Rect &left_crop_rect, Rect &right_crop_rect, int dst_width)
{
    cv::UMat left_umat, right_umat;
    cv::Mat left_mat, right_mat;
    cv::_InputArray left_img, right_img;

    if (!get_crop_image (left_buf, left_crop_rect, left_umat)
            || !get_crop_image (right_buf, right_crop_rect, right_umat))
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

    detect_and_match_cluster (left_img, right_img, left_crop_rect, right_crop_rect,
                              _mean_offset, _mean_offset_y, _x_offset, _y_offset);

#if XCAM_CV_FM_DEBUG
    XCAM_ASSERT (_fm_idx >= 0);

    char frame_str[64] = {'\0'};
    std::snprintf (frame_str, 64, "frame:%d", _frame_num);
    char fm_idx_str[64] = {'\0'};
    std::snprintf (fm_idx_str, 64, "fm_idx:%d", _fm_idx);

    char img_name[256] = {'\0'};
    std::snprintf (img_name, 256, "fm_in_stitch_area_%d_%d_0.jpg", _frame_num, _fm_idx);
    debug_write_image (left_buf, left_crop_rect, img_name, frame_str, fm_idx_str);

    std::snprintf (img_name, 256, "fm_in_stitch_area_%d_%d_1.jpg", _frame_num, _fm_idx);
    debug_write_image (right_buf, right_crop_rect, img_name, frame_str, fm_idx_str);

    XCAM_LOG_INFO ("FeatureMatch(idx:%d): frame number:%d done", _fm_idx, _frame_num);
    _frame_num++;
#endif

    XCAM_UNUSED (dst_width);
}

}
