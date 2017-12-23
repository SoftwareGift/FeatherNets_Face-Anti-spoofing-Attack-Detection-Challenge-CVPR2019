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

#include "cv_image_deblurring.h"

namespace XCam {


CVImageDeblurring::CVImageDeblurring ()
    : CVBaseClass ()
{
    _helper = new CVImageProcessHelper ();
    _sharp = new CVImageSharp ();
    _edgetaper = new CVEdgetaper ();
    _wiener = new CVWienerFilter ();
}

void
CVImageDeblurring::set_config (CVIDConfig config)
{
    _config = config;
}

CVIDConfig
CVImageDeblurring::get_config ()
{
    return _config;
}

void
CVImageDeblurring::crop_border (cv::Mat &thresholded)
{
    int top = 0;
    int left = 0;
    int right = 0;
    int bottom = 0;
    for (int i = 0; i < thresholded.rows; i++)
    {
        for (int j = 0; j < thresholded.cols; j++)
        {
            if (thresholded.at<unsigned char>(i , j) == 255)
            {
                top = i;
                break;
            }
        }
        if (top)
            break;
    }

    for (int i = thresholded.rows - 1; i > 0; i--)
    {
        for (int j = 0; j < thresholded.cols; j++)
        {
            if (thresholded.at<unsigned char>(i , j) == 255)
            {
                bottom = i;
                break;
            }
        }
        if (bottom)
            break;
    }

    for (int i = 0; i < thresholded.cols; i++)
    {
        for (int j = 0; j < thresholded.rows; j++)
        {
            if (thresholded.at<unsigned char>(j , i) == 255)
            {
                left = i;
                break;
            }
        }
        if (left)
            break;
    }

    for (int i = thresholded.cols - 1; i > 0; i--)
    {
        for (int j = 0; j < thresholded.rows; j++)
        {
            if (thresholded.at<unsigned char>(j, i) == 255)
            {
                right = i;
                break;
            }
        }
        if (right)
            break;
    }
    thresholded = thresholded (cv::Rect(left, top, right - left, bottom - top));
}

int
CVImageDeblurring::estimate_kernel_size (const cv::Mat &image)
{
    int kernel_size = 0;
    cv::Mat thresholded;
    cv::Mat dst;
    cv::Laplacian (image, dst, -1, 3, 1, 0, cv::BORDER_CONSTANT);
    dst.convertTo (dst, CV_32FC1);
    cv::filter2D (dst, thresholded, -1, dst, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);

    for (int i = 0; i < 10; i++)
    {
        cv::Mat thresholded_new;
        double min_val;
        double max_val;
        cv::minMaxLoc (thresholded, &min_val, &max_val);
        cv::threshold (thresholded, thresholded, round(max_val / 3.5), 255, cv::THRESH_BINARY);
        thresholded.convertTo (thresholded, CV_8UC1);
        crop_border (thresholded);
        if (thresholded.rows < 3)
        {
            break;
        }
        int filter_size = (int)(std::max(3, ((thresholded.rows + thresholded.cols) / 2) / 10));
        if (!(filter_size & 1))
        {
            filter_size++;
        }
        cv::Mat filter = cv::Mat::ones (filter_size, filter_size, CV_32FC1) / (float)(filter_size * filter_size - 1);
        filter.at<float> (filter_size / 2, filter_size / 2) = 0;
        cv::filter2D (thresholded, thresholded_new, -1, filter, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
        kernel_size = (thresholded_new.rows + thresholded_new.cols) / 2;
        if (!(kernel_size & 1))
        {
            kernel_size++;
        }
        thresholded = thresholded_new.clone();
    }
    return kernel_size;
}

void
CVImageDeblurring::blind_deblurring (const cv::Mat &blurred, cv::Mat &deblurred, cv::Mat &kernel, int kernel_size, float noise_power, bool use_edgetaper)
{
    cv::Mat gray_blurred;
    cv::cvtColor (blurred, gray_blurred, CV_BGR2GRAY);
    if (noise_power < 0)
    {
        cv::Mat median_blurred;
        medianBlur (gray_blurred, median_blurred, 3);
        noise_power = 1.0f / _helper->get_snr (gray_blurred, median_blurred);
        XCAM_LOG_DEBUG ("estimated inv snr %f", noise_power);
    }
    if (kernel_size < 0)
    {
        kernel_size = estimate_kernel_size (gray_blurred);
        XCAM_LOG_DEBUG ("estimated kernel size %d", kernel_size);
    }
    if (use_edgetaper) {
        XCAM_LOG_DEBUG ("edgetaper will be used");
    }
    else {
        XCAM_LOG_DEBUG ("edgetaper will not be used");
    }
    std::vector<cv::Mat> blurred_rgb (3);
    cv::split (blurred, blurred_rgb);
    std::vector<cv::Mat> deblurred_rgb (3);
    cv::Mat result_deblurred;
    cv::Mat result_kernel;
    blind_deblurring_one_channel (gray_blurred, result_kernel, kernel_size, noise_power);
    for (int i = 0; i < 3; i++)
    {
        cv::Mat input;
        if (use_edgetaper)
        {
            _edgetaper->edgetaper (blurred_rgb[i], result_kernel, input);
        }
        else
        {
            input = blurred_rgb[i].clone ();
        }
        _wiener->wiener_filter (input, result_kernel, deblurred_rgb[i], noise_power);
        _helper->apply_constraints (deblurred_rgb[i], 0);
    }
    cv::merge (deblurred_rgb, result_deblurred);
    result_deblurred.convertTo (result_deblurred, CV_8UC3);
    fastNlMeansDenoisingColored (result_deblurred, deblurred, 3, 3, 7, 21);
    kernel = result_kernel.clone ();
}

void
CVImageDeblurring::blind_deblurring_one_channel (const cv::Mat &blurred, cv::Mat &kernel, int kernel_size, float noise_power)
{
    cv::Mat kernel_current = cv::Mat::zeros (kernel_size, kernel_size, CV_32FC1);
    cv::Mat deblurred_current = _helper->erosion (blurred, 2, 0);
    float sigmar = 20;
    for (int i = 0; i < _config.iterations; i++)
    {
        cv::Mat sharpened = _sharp->sharp_image_gray (deblurred_current, sigmar);
        _wiener->wiener_filter (blurred, sharpened.clone (), kernel_current, noise_power);
        kernel_current = kernel_current (cv::Rect (0, 0, kernel_size, kernel_size));
        double min_val;
        double max_val;
        cv::minMaxLoc (kernel_current, &min_val, &max_val);
        _helper->apply_constraints (kernel_current, (float)max_val / 20);
        _helper->normalize_weights (kernel_current);
        _wiener->wiener_filter (blurred, kernel_current.clone(), deblurred_current, noise_power);
        _helper->apply_constraints (deblurred_current, 0);
        sigmar *= 0.9;
    }
    kernel = kernel_current.clone ();
}

}
