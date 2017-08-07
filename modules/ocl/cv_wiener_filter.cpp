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
    cv::Mat yFT[2];
    _helpers->compute_dft (blurred_image, yFT);

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

    float padded_re;
    float padded_im;
    float padded_abs;
    float denominator;
    std::complex<float> numerator;

    for (int i = 0; i < padded.rows; i++)
    {
        for (int j = 0; j < padded.cols; j++)
        {
            padded_re = padded_ft[0].at<float>(i, j);
            padded_im = padded_ft[1].at<float>(i, j);
            padded_abs = padded_re * padded_re + padded_im * padded_im;
            denominator = noise_power + padded_abs;
            numerator = std::complex<float>(padded_re, -padded_im) * std::complex<float>(yFT[0].at<float>(i, j), yFT[1].at<float>(i, j));
            unknown_ft[0].at<float>(i, j) = numerator.real() / denominator;
            unknown_ft[1].at<float>(i, j) = numerator.imag() / denominator;
        }
    }
    _helpers->compute_idft (unknown_ft, temp_unknown);
    rotate (temp_unknown, temp_unknown);
    unknown = temp_unknown.clone();
}

}
