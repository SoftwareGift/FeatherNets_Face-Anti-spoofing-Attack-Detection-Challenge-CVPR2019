/*
 * cv_capi_feature_match.cpp - optical flow feature match
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
 * Author: Zong Wei <wei.zong@intel.com>
 */

#include "cv_capi_feature_match.h"

#define XCAM_CV_CAPI_FM_DEBUG 0

namespace XCam {

CVCapiFeatureMatch::CVCapiFeatureMatch ()
    : FeatureMatch ()
{
}

bool
CVCapiFeatureMatch::get_crop_image (
    const SmartPtr<VideoBuffer> &buffer, const Rect &crop_rect, std::vector<char> &crop_image, CvMat &img)
{
    VideoBufferInfo info = buffer->get_video_info ();

    uint8_t* image_buffer = buffer->map();
    int offset = info.strides[NV12PlaneYIdx] * crop_rect.pos_y + crop_rect.pos_x;

    crop_image.resize (crop_rect.width * crop_rect.height);
    for (int i = 0; i < crop_rect.height; i++) {
        for (int j = 0; j < crop_rect.width; j++) {
            crop_image[i * crop_rect.width + j] =
                image_buffer[offset + i * info.strides[NV12PlaneYIdx] + j];
        }
    }

    img = cvMat (crop_rect.height, crop_rect.width, CV_8UC1, (void*)&crop_image[0]);

    return true;
}

void
CVCapiFeatureMatch::add_detected_data (CvArr* image, std::vector<CvPoint2D32f> &corners)
{
    std::vector<CvPoint2D32f> keypoints;

    int found_num = 300;
    double quality = 0.01;
    double min_dist = 5;

    corners.resize (found_num);
    CvPoint2D32f* corner_points = &corners[0];

    cvGoodFeaturesToTrack (image, NULL, NULL, corner_points, &found_num, quality, min_dist);
    XCAM_ASSERT (found_num <= 300);

#if XCAM_CV_CAPI_FM_DEBUG
    XCAM_LOG_INFO ("FeatureMatch(idx:%d): detected corners:%d, reserved size:%d", _fm_idx, found_num, (int)corners.size ());
#endif
    if (found_num < (int)corners.size ())
        corners.resize (found_num);
}

void
CVCapiFeatureMatch::get_valid_offsets (
    std::vector<CvPoint2D32f> &corner0, std::vector<CvPoint2D32f> &corner1,
    std::vector<char> &status, std::vector<float> &error,
    std::vector<float> &offsets, float &sum, int &count,
    CvArr* image, CvSize &img0_size)
{
    count = 0;
    sum = 0.0f;

    for (uint32_t i = 0; i < status.size (); ++i) {
        if (!status[i])
            continue;

#if XCAM_CV_CAPI_FM_DEBUG
        cv::Mat mat = cv::cvarrToMat (image);
        cv::Point start = cv::Point (corner0[i].x, corner0[i].y);
        cv::circle (mat, start, 2, cv::Scalar(255), 2);
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

#if XCAM_CV_CAPI_FM_DEBUG
        cv::line (mat, start, cv::Point(corner1[i].x + img0_size.width, corner1[i].y), cv::Scalar(255), 2);
#else
        XCAM_UNUSED (image);
        XCAM_UNUSED (img0_size);
#endif
    }
}

void
CVCapiFeatureMatch::calc_of_match (
    CvArr* image0, CvArr* image1, std::vector<CvPoint2D32f> &corner0, std::vector<CvPoint2D32f> &corner1,
    std::vector<char> &status, std::vector<float> &error)
{
    CvMat debug_image;
    CvSize img0_size = cvSize(((CvMat*)image0)->width, ((CvMat*)image0)->height);
    XCAM_ASSERT (img0_size.height == ((CvMat*)image1)->height);
    XCAM_UNUSED (image1);

    std::vector<float> offsets;
    float offset_sum = 0.0f;
    int count = 0;
    float mean_offset = 0.0f;
    float last_mean_offset = _mean_offset;
    offsets.reserve (corner0.size ());

#if XCAM_CV_CAPI_FM_DEBUG
    CvSize img1_size = cvSize(((CvMat*)image1)->width, ((CvMat*)image1)->height);
    cv::Mat mat;
    mat.create (img0_size.height, img0_size.width + img1_size.width, ((CvMat*)image0)->type);
    debug_image = cvMat (img0_size.height, img0_size.width + img1_size.width, ((CvMat*)image0)->type, mat.ptr());
    cv::cvarrToMat(image0, true).copyTo (mat (cv::Rect(0, 0, img0_size.width, img0_size.height)));
    cv::cvarrToMat(image1, true).copyTo (mat (cv::Rect(img0_size.width, 0, img1_size.width, img1_size.height)));
#endif

    get_valid_offsets (corner0, corner1, status, error,
                       offsets, offset_sum, count, &debug_image, img0_size);

#if XCAM_CV_CAPI_FM_DEBUG
    XCAM_LOG_INFO ("FeatureMatch(idx:%d): valid offsets:%d", _fm_idx, offsets.size ());
    char file_name[256] = {'\0'};
    std::snprintf (file_name, 256, "fm_optical_flow_%d_%d.jpg", _frame_num, _fm_idx);
    cv::imwrite (file_name, mat);
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
CVCapiFeatureMatch::detect_and_match (CvArr* img_left, CvArr* img_right)
{
    std::vector<float> err;
    std::vector<char> status;
    std::vector<CvPoint2D32f> corner_left, corner_right;

    CvSize win_size = cvSize (41, 41);

    add_detected_data (img_left, corner_left);
    int count = corner_left.size ();
    if (corner_left.empty ()) {
        return;
    }

    // find the corresponding points in img_right
    corner_right.resize (count);
    status.resize (count);
    err.resize (count);

    CvPoint2D32f* corner_points1 = &corner_left[0];
    CvPoint2D32f* corner_points2 = &corner_right[0];
    char* optflow_status = &status[0];
    float* optflow_errs = &err[0];

    cvCalcOpticalFlowPyrLK (
        img_left, img_right, 0, 0, corner_points1, corner_points2, count, win_size, 3,
        optflow_status, optflow_errs, cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10, 0.01f), 0);

#if XCAM_CV_CAPI_FM_DEBUG
    XCAM_LOG_INFO ("FeatureMatch(idx:%d): matched corners:%d", _fm_idx, count);
#endif

    calc_of_match (img_left, img_right, corner_left, corner_right, status, err);

#if XCAM_CV_CAPI_FM_DEBUG
    XCAM_LOG_INFO ("FeatureMatch(idx:%d): x_offset:%0.2f", _fm_idx, _x_offset);
#endif
}

void
CVCapiFeatureMatch::feature_match (
    const SmartPtr<VideoBuffer> &left_buf, const SmartPtr<VideoBuffer> &right_buf)
{
    XCAM_ASSERT (_left_rect.width && _left_rect.height);
    XCAM_ASSERT (_right_rect.width && _right_rect.height);

    CvMat left_img, right_img;
    if (!get_crop_image (left_buf, _left_rect, _left_crop_image, left_img)
            || !get_crop_image (right_buf, _right_rect, _right_crop_image, right_img))
        return;

    detect_and_match ((CvArr*)(&left_img), (CvArr*)(&right_img));

#if XCAM_CV_CAPI_FM_DEBUG
    XCAM_ASSERT (_fm_idx >= 0);

    char frame_str[64] = {'\0'};
    std::snprintf (frame_str, 64, "frame:%d", _frame_num);
    char fm_idx_str[64] = {'\0'};
    std::snprintf (fm_idx_str, 64, "fm_idx:%d", _fm_idx);

    char img_name[256] = {'\0'};
    std::snprintf (img_name, 256, "fm_in_stitch_area_%d_%d_0.jpg", _frame_num, _fm_idx);
    write_image (left_buf, _left_rect, img_name, frame_str, fm_idx_str);
    std::snprintf (img_name, 256, "fm_in_stitch_area_%d_%d_1.jpg", _frame_num, _fm_idx);
    write_image (right_buf, _right_rect, img_name, frame_str, fm_idx_str);

    XCAM_LOG_INFO ("FeatureMatch(idx:%d): frame number:%d done", _fm_idx, _frame_num);

    _frame_num++;
#endif
}

}
