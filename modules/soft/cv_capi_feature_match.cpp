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

namespace XCam {

CVCapiFeatureMatch::CVCapiFeatureMatch ()
    : FeatureMatch()
{
}

bool
CVCapiFeatureMatch::get_crop_image (
    SmartPtr<VideoBuffer> buffer, Rect crop_rect, std::vector<char> &crop_image, CvMat &img)
{
    VideoBufferInfo info = buffer->get_video_info ();

    uint8_t* image_buffer = buffer->map();
    int offset = info.strides[CLNV12PlaneY] * crop_rect.pos_y + crop_rect.pos_x;

    crop_image.resize (crop_rect.width * crop_rect.height);
    for (int i = 0; i < crop_rect.height; i++) {
        for (int j = 0; j < crop_rect.width; j++) {
            crop_image[i * crop_rect.width + j] =
                image_buffer[offset + i * info.strides[CLNV12PlaneY] + j];
        }
    }

    img = cvMat (crop_rect.height, crop_rect.width, CV_8UC1, (void*)&crop_image[0]);

    return true;
}

void
CVCapiFeatureMatch::add_detected_data (
    CvArr* image, std::vector<CvPoint2D32f> &corners)
{
    std::vector<CvPoint2D32f> keypoints;

    int found_num = 300;
    double quality = 0.01;
    double min_dist = 5;

    corners.resize (found_num);
    CvPoint2D32f* corner_points = &corners[0];

    cvGoodFeaturesToTrack (image, NULL, NULL, corner_points, &found_num, quality, min_dist);
    XCAM_ASSERT (found_num <= 300);
    if (found_num < (int)corners.size ()) {
        XCAM_LOG_DEBUG ("Detedt corners:%d, less than reserved size:%d\n", found_num, (int)corners.size ());
        corners.resize (found_num);
    }
}

void
CVCapiFeatureMatch::get_valid_offsets (
    CvArr* image, CvSize img0_size,
    std::vector<CvPoint2D32f> corner0, std::vector<CvPoint2D32f> corner1,
    std::vector<char> status, std::vector<float> error,
    std::vector<float> &offsets, float &sum, int &count)
{
    count = 0;
    sum = 0.0f;
    for (uint32_t i = 0; i < status.size (); ++i) {
        if (!status[i] || error[i] > 24)
            continue;
        if (fabs(corner0[i].y - corner1[i].y) >= 8)
            continue;

        float offset = corner1[i].x - corner0[i].x;
        sum += offset;
        ++count;
        offsets.push_back (offset);

        XCAM_UNUSED (image);
        XCAM_UNUSED (img0_size);
    }
}

void
CVCapiFeatureMatch::calc_of_match (
    CvArr* image0, CvArr* image1,
    std::vector<CvPoint2D32f> corner0, std::vector<CvPoint2D32f> corner1,
    std::vector<char> &status, std::vector<float> &error,
    int &last_count, float &last_mean_offset, float &out_x_offset)
{
    CvSize img0_size = cvSize(((CvMat*)image0)->width, ((CvMat*)image0)->height);
    CvSize img1_size = cvSize(((CvMat*)image1)->width, ((CvMat*)image1)->height);
    XCAM_ASSERT (img0_size.height == img1_size.height);

    std::vector<float> offsets;
    float offset_sum = 0.0f;
    int count = 0;
    float mean_offset = 0.0f;
    offsets.reserve (corner0.size ());
    get_valid_offsets (NULL, img0_size, corner0, corner1, status, error,
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
}

void
CVCapiFeatureMatch::detect_and_match (
    CvArr* img_left, CvArr* img_right, Rect &crop_left, Rect &crop_right,
    int &valid_count, float &mean_offset, float &x_offset, int dst_width)
{
    std::vector<float> err;
    std::vector<char> status;
    std::vector<CvPoint2D32f> corner_left, corner_right;

    CvSize win_size = cvSize (5, 5);

    add_detected_data (img_left, corner_left);
    int count = corner_left.size ();
    if (corner_left.empty ()) {
        return;
    }

    // find the corresponding points in img_right
    err.resize (count);
    status.resize (count);
    corner_left.resize (count);
    corner_right.resize (count);

    float* optflow_errs = &err[0];
    char* optflow_status = &status[0];
    CvPoint2D32f* corner_points1 = &corner_left[0];
    CvPoint2D32f* corner_points2 = &corner_right[0];

    for (size_t i = 0; i < (size_t)count; ++i) {
        corner_points1[i] = corner_left[i];
    }

    cvCalcOpticalFlowPyrLK (
        img_left, img_right, 0, 0, corner_points1, corner_points2, count, win_size, 3,
        optflow_status, optflow_errs, cvTermCriteria(CV_TERMCRIT_ITER, 40, 0.1), 0 );

    corner_right.reserve (count);
    status.reserve (count);
    err.reserve (count);
    for (size_t i = 0; i < (size_t)count; ++i) {
        CvPoint2D32f &kp = corner_points2[i];
        corner_right.push_back (kp);
        status.push_back (optflow_status[i]);
        err.push_back (optflow_errs[i]);
    }

    calc_of_match (img_left, img_right, corner_left, corner_right,
                   status, err, valid_count, mean_offset, x_offset);

    adjust_stitch_area (dst_width, x_offset, crop_left, crop_right);
}

void
CVCapiFeatureMatch::optical_flow_feature_match (
    SmartPtr<VideoBuffer> left_buf, SmartPtr<VideoBuffer> right_buf,
    Rect &left_crop_rect, Rect &right_crop_rect, int dst_width)
{
    CvMat left_img, right_img;

    if (!get_crop_image (left_buf, left_crop_rect, _left_crop_image, left_img)
            || !get_crop_image (right_buf, right_crop_rect, _right_crop_image, right_img))
        return;

    detect_and_match ((CvArr*)(&left_img), (CvArr*)(&right_img), left_crop_rect, right_crop_rect,
                      _valid_count, _mean_offset, _x_offset, dst_width);
}

}
