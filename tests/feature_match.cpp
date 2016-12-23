/*
 * feature_match.cpp - optical flow feature match
 *
 *  Copyright (c) 2016 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * 	 http://www.apache.org/licenses/LICENSE-2.0
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

#include <base/xcam_common.h>
#include <base/xcam_buffer.h>
#include <dma_video_buffer.h>
#include <smartptr.h>

#include <cl_context.h>
#include <cl_device.h>
#include <cl_memory.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>

using namespace std;
using namespace cv;
using namespace XCam;

#define XCAM_OF_DEBUG 0
#define XCAM_OF_DRAW_SCALE 2

static int sitch_min_width = 64;
static int corner_min_num = 16;
static float max_offset = 8.0f;

typedef struct {
    bool valid;
    float data;
} OFOffset;

void
init_opencv_ocl (SmartPtr<CLContext> context)
{
    static bool is_ocl_inited = false;

    if (!is_ocl_inited) {
        cl_platform_id platform_id = CLDevice::instance()->get_platform_id ();
        char *platform_name = CLDevice::instance()->get_platform_name ();
        cl_device_id device_id = CLDevice::instance()->get_device_id ();
        cl_context context_id = context->get_context_id ();
        ocl::attachContext (platform_name, platform_id, context_id, device_id);

        is_ocl_inited = true;
    }
}

bool
convert_to_mat (SmartPtr<CLContext> context, SmartPtr<DrmBoBuffer> buffer, Mat &image)
{
    SmartPtr<CLBuffer> cl_buffer = new CLVaBuffer (context, buffer);
    VideoBufferInfo info = buffer->get_video_info ();
    cl_mem cl_mem_id = cl_buffer->get_mem_id ();

    UMat umat;
    ocl::convertFromBuffer (cl_mem_id, info.strides[0], info.height * 3 / 2, info.width, CV_8U, umat);
    if (umat.empty ()) {
        XCAM_LOG_ERROR ("convert bo buffer to UMat failed");
        return false;
    }

    Mat mat;
    umat.copyTo (mat);
    if (mat.empty ()) {
        XCAM_LOG_ERROR ("copy UMat to Mat failed");
        return false;
    }

    cvtColor (mat, image, COLOR_YUV2BGR_NV12);
    return true;
}

static void
add_detected_data (Mat image, Ptr<Feature2D> detector, vector<Point2f> &corners)
{
    vector<KeyPoint> keypoints;
    detector->detect (image, keypoints);
    corners.reserve (corners.size () + keypoints.size ());
    for (size_t i = 0; i < keypoints.size (); ++i) {
        KeyPoint &kp = keypoints[i];
        corners.push_back (kp.pt);
    }
}

static Mat
draw_match_optical_flow (
    Mat image0, Mat image1,
    vector<Point2f> corner0, vector<Point2f> corner1,
    vector<uchar> &status, vector<float> &error,
    OFOffset &out_x_offset)
{
    Mat out_image;
    Size img0_size = image0.size ();
    Size img1_size = image1.size ();
    XCAM_ASSERT (img0_size.height == img1_size.height);
    Size size (img0_size.width + img1_size.width, img0_size.height);

    out_image.create (size, image0.type ());
    image0.copyTo (out_image (Rect(0, 0, img0_size.width, img0_size.height)));
    image1.copyTo (out_image (Rect(img0_size.width, 0, img1_size.width, img1_size.height)));

#if XCAM_OF_DEBUG
    Size scale_size = size * XCAM_OF_DRAW_SCALE;
    resize (out_image, out_image, scale_size, 0, 0);
#endif

    float x_offset_sum = 0.0f;
    vector<float> offsets;
    int count = 0;

    offsets.reserve (corner0.size ());
    for (uint32_t i = 0; i < status.size (); ++i) {
#if XCAM_OF_DEBUG
        Point start = Point(corner0[i]) * XCAM_OF_DRAW_SCALE;
        circle (out_image, start, 4, Scalar(255, 0, 0), XCAM_OF_DRAW_SCALE);
#endif
        if (!status[i] || error[i] > 30)
            continue;

        if (fabs(corner0[i].y - corner1[i].y) > 8)
            continue;

        float offset = (corner1[i].x - corner0[i].x);
        x_offset_sum += offset;
        ++count;
        offsets.push_back (offset);

#if XCAM_OF_DEBUG
        Point end = (Point(corner1[i]) + Point (img0_size.width, 0))* XCAM_OF_DRAW_SCALE;
        line(out_image, start, end, Scalar(0, 0,255), XCAM_OF_DRAW_SCALE);
#endif
    }

    float mean_offset = max_offset + 1.0f;
    if (count >= corner_min_num) {
        float mean_offset = x_offset_sum / count;
        XCAM_LOG_INFO ("X-axis avg offset : %.2f, (count:%d)", mean_offset, count);
        for (int try_times = 1; try_times < 4; ++try_times) {
            x_offset_sum = 0.0f;
            int recur_count = 0;
            for (size_t i = 0; i < offsets.size (); ++i) {
                if (fabs(offsets[i] - mean_offset) > 8.0f)
                    continue;
                x_offset_sum += offsets[i];
                ++recur_count;
            }

            if (recur_count < corner_min_num)
                break;

            mean_offset = x_offset_sum / recur_count;
            XCAM_LOG_INFO ("X-axis mean offset:%.2f, (%d times, count:%d)", mean_offset, try_times, recur_count);
        }
    }

    if (count >= corner_min_num && mean_offset <= max_offset) {
        out_x_offset.valid = true;
        out_x_offset.data = mean_offset;
    } else {
        out_x_offset.valid = false;
    }
    return out_image;
}

static void
adjust_stitch_area (int dst_width, Rect &stitch0, Rect &stitch1)
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

void
optical_flow_feature_match (
    SmartPtr<CLContext> context, int dst_width,
    SmartPtr<DrmBoBuffer> buf0, SmartPtr<DrmBoBuffer> buf1,
    Rect &image0_crop_left, Rect &image0_crop_right,
    Rect &image1_crop_left, Rect &image1_crop_right,
    const char *input_name, int frame_num)
{
    Mat image0, image1;
    Mat image0_left, image0_right, image1_left, image1_right;
    Mat image0_left_rgb, image0_right_rgb, image1_left_rgb, image1_right_rgb;
    vector<Point2f> corner0_left, corner0_right, corner1_left, corner1_right;
    Mat out_image0, out_image1;
    OFOffset x_offset0, x_offset1;

    if (!convert_to_mat (context, buf0, image0) || !convert_to_mat (context, buf1, image1))
        return;

    image0_left_rgb = image0 (image0_crop_left);
    cvtColor (image0_left_rgb, image0_left, COLOR_BGR2GRAY);
    image0_right_rgb = image0 (image0_crop_right);
    cvtColor (image0_right_rgb, image0_right, COLOR_BGR2GRAY);

    image1_left_rgb = image1 (image1_crop_left);
    cvtColor (image1_left_rgb, image1_left, COLOR_BGR2GRAY);
    image1_right_rgb = image1 (image1_crop_right);
    cvtColor (image1_right_rgb, image1_right, COLOR_BGR2GRAY);

    Ptr<Feature2D> gft_detector, orb_detector;
    gft_detector = GFTTDetector::create (300, 0.01, 5, 5, false);
    orb_detector = ORB::create (200, 1.5, 2, 9);

    add_detected_data (image0_left, gft_detector, corner0_left);
    add_detected_data (image0_left, orb_detector, corner0_left);
    add_detected_data (image0_right, gft_detector, corner0_right);
    add_detected_data (image0_right, orb_detector, corner0_right);

    vector<float> err0, err1;
    vector<uchar> status0, status1;
	calcOpticalFlowPyrLK (
        image0_left, image1_right, corner0_left, corner1_right,
        status0, err0, Size(5,5), 3,
        TermCriteria (TermCriteria::COUNT+TermCriteria::EPS, 10, 0.01));
    calcOpticalFlowPyrLK (
        image0_right, image1_left, corner0_right, corner1_left,
        status1, err1, Size(5,5), 3,
        TermCriteria (TermCriteria::COUNT+TermCriteria::EPS, 10, 0.01));

    out_image0 = draw_match_optical_flow (image0_left_rgb, image1_right_rgb,
                                          corner0_left, corner1_right, status0, err0, x_offset0);
    if (x_offset0.valid) {
        image0_crop_left.x += x_offset0.data;
        adjust_stitch_area (dst_width, image1_crop_right, image0_crop_left);
        XCAM_LOG_INFO (
            "Stiching area 0: image0_left_area(x:%d, width:%d), image1_right_area(x:%d, width:%d)",
            image0_crop_left.x, image0_crop_left.width, image1_crop_right.x, image1_crop_right.width);
    }

    out_image1 = draw_match_optical_flow (image0_right_rgb, image1_left_rgb,
                                          corner0_right, corner1_left, status1, err1, x_offset1);
    if (x_offset1.valid) {
        image0_crop_right.x += x_offset1.data;
        adjust_stitch_area (dst_width, image0_crop_right, image1_crop_left);
        XCAM_LOG_INFO (
            "Stiching area 1: image0_right_area(x:%d, width:%d), image1_left_area(x:%d, width:%d)",
            image0_crop_right.x, image0_crop_right.width, image1_crop_left.x, image1_crop_left.width);
    }

#if XCAM_OF_DEBUG
    char file_name[1024];
    char *tmp_name = strdup (input_name);
    char *prefix = strtok (tmp_name, ".");
    snprintf (file_name, 1024, "%s_%d_OF_stitching_0.jpg", prefix, frame_num);
    imwrite (file_name, out_image0);
    XCAM_LOG_INFO ("write feature match: %s", file_name);

    snprintf (file_name, 1024, "%s_%d_OF_stitching_1.jpg", prefix, frame_num);
    imwrite (file_name, out_image1);
    XCAM_LOG_INFO ("write feature match: %s", file_name);
    free (tmp_name);
#else
    XCAM_UNUSED (input_name);
    XCAM_UNUSED (frame_num);
#endif
}

