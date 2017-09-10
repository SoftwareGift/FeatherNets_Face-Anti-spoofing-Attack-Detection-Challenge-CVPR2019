/*
 * cv_image_deblurring.cpp - iterative blind deblurring
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
 * Author: Andrey Parfenov <a1994ndrey@gmail.com>
 * Author: Wind Yuan <feng.yuan@intel.com>
 */

#include "cv_wiener_filter.h"

namespace XCam {


CVWienerFilter::CVWienerFilter ()
    : CVBaseClass()
{
    _helpers = new CVImageProcessHelper();
}

void
CVWienerFilter::rotate (cv::Mat &src, cv::Mat &dst)
{
    int cx = src.cols >> 1;
    int cy = src.rows >> 1;
    cv::Mat tmp;
    tmp.create(src.size (), src.type ());
    src(cv::Rect(0, 0, cx, cy)).copyTo(tmp(cv::Rect(cx, cy, cx, cy)));
    src(cv::Rect(cx, cy, cx, cy)).copyTo(tmp(cv::Rect(0, 0, cx, cy)));
    src(cv::Rect(cx, 0, cx, cy)).copyTo(tmp(cv::Rect(0, cy, cx, cy)));
    src(cv::Rect(0, cy, cx, cy)).copyTo(tmp(cv::Rect(cx, 0, cx, cy)));
    dst = tmp.clone();
}

void
CVWienerFilter::wiener_filter (const cv::Mat &blurred_image, const cv::Mat &known, cv::Mat &unknown, float noise_power)
{
    int image_w = blurred_image.size().width;
    int image_h = blurred_image.size().height;
    cv::Mat y_ft[2];
    _helpers->compute_dft (blurred_image, y_ft);

    cv::Mat padded = cv::Mat::zeros(image_h, image_w, CV_32FC1);
    int padx = padded.cols - known.cols;
    int pady = padded.rows - known.rows;
    cv::copyMakeBorder (known, padded, pady / 2, pady - pady / 2, padx / 2, padx - padx / 2, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    cv::Mat padded_ft[2];
    _helpers->compute_dft (padded, padded_ft);

    cv::Mat temp_unknown;
    cv::Mat unknown_ft[2];
    unknown_ft[0] = cv::Mat::zeros(image_h, image_w, CV_32FC1);
    unknown_ft[1] = cv::Mat::zeros(image_h, image_w, CV_32FC1);

    cv::Mat denominator;
    cv::Mat padded_re;
    cv::Mat padded_im;
    cv::pow (padded_ft[0], 2, padded_re);
    cv::pow (padded_ft[1], 2, padded_im);
    denominator = padded_re + padded_im + cv::Scalar (noise_power);

    cv::Mat numerator_real;
    cv::Mat numerator_im;
    cv::Mat first_term;
    cv::Mat second_term;
    first_term = padded_ft[0].mul (y_ft[0]);
    second_term = padded_ft[1].mul (y_ft[1]);
    numerator_real = first_term + second_term;
    first_term = padded_ft[0].mul (y_ft[1]);
    second_term = padded_ft[1].mul (y_ft[0]);
    numerator_im = first_term - second_term;
    cv::divide (numerator_real, denominator, unknown_ft[0]);
    cv::divide (numerator_im, denominator, unknown_ft[1]);

    _helpers->compute_idft (unknown_ft, temp_unknown);
    rotate (temp_unknown, temp_unknown);
    unknown = temp_unknown.clone();
}

}
