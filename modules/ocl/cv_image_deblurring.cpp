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
    : CVBaseClass()
{
    _helper = new CVImageProcessHelper();
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
CVImageDeblurring::normalized_autocorrelation (const cv::Mat &psf, cv::Mat &auto_correlation_psf)
{
    cv::Mat correlation;
    cv::copyMakeBorder (psf, auto_correlation_psf, psf.cols - 1, 0, psf.rows - 1, 0, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    cv::filter2D (auto_correlation_psf, correlation, -1, psf, cv::Point(0, 0), 0, cv::BORDER_CONSTANT);
    cv::normalize (correlation, correlation, 0, 1.0f, cv::NORM_MINMAX);
    auto_correlation_psf = correlation.clone ();
}

void
CVImageDeblurring::create_weights (cv::Mat &coefficients, const cv::Mat &psf)
{
    int psfr_last = psf.rows - 1;
    int psfc_last = psf.cols - 1;

    cv::Mat auto_correlation_psf;
    normalized_autocorrelation (psf, auto_correlation_psf);

    for (int i = 0; i < coefficients.rows; i++)
    {
        for (int j = 0; j < coefficients.cols; j++) {
            if (i < psfr_last)
            {
                if (j < psfc_last)
                    coefficients.at<float>(i, j) = auto_correlation_psf.at<float>(i, j);
                else if (psfc_last <= j && j < (coefficients.cols - psfc_last))
                    coefficients.at<float>(i, j) = auto_correlation_psf.at<float>(i, psfc_last);
                else
                    coefficients.at<float>(i, j) = auto_correlation_psf.at<float>(i, j - (coefficients.cols - 2 * psfc_last) + 1);
            }
            else if (psfr_last <= i && i < (coefficients.rows - psfr_last))
            {
                if (j < psfc_last)
                    coefficients.at<float>(i, j) = auto_correlation_psf.at<float>(psfr_last, j);
                else if (psfc_last <= j && j < (coefficients.cols - psfc_last))
                    coefficients.at<float>(i, j) = 1.0f;
                else
                    coefficients.at<float>(i, j) = auto_correlation_psf.at<float>(psfr_last, j - (coefficients.cols - 2 * psfc_last) + 1);
            }
            else
            {
                if (j < psfc_last)
                    coefficients.at<float>(i, j) = auto_correlation_psf.at<float>(i - (coefficients.rows - 2 * psfr_last) + 1, j);
                else if (psfc_last <= j && j < (coefficients.cols - psfc_last))
                    coefficients.at<float>(i, j) = auto_correlation_psf.at<float>(i - (coefficients.rows - 2 * psfr_last) + 1, psfc_last);
                else
                    coefficients.at<float>(i, j) = auto_correlation_psf.at<float>(i - (coefficients.rows - 2 * psfr_last) + 1, j - (coefficients.cols - 2 * psfc_last) + 1);
            }
        }
    }
}

cv::Mat
CVImageDeblurring::edgetaper (const cv::Mat &img, const cv::Mat &psf)
{
    cv::Mat blurred = cv::Mat::zeros (img.rows, img.cols, CV_32FC1);
    cv::filter2D (img, blurred, CV_32F, psf, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);

    cv::Mat coefficients = cv::Mat::zeros (img.rows, img.cols, CV_32FC1);
    create_weights (coefficients, psf);
    cv::Mat result = img.clone ();
    result.convertTo (result, CV_32FC1);
    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
            if (coefficients.at<float>(i, j) != 1.0f)
            {
                result.at<float>(i, j) = img.at<unsigned char>(i, j) * coefficients.at<float>(i, j) +
                                         blurred.at<float>(i, j) * (1.0f - coefficients.at<float>(i, j));
            }
        }
    }
    return result;
}

cv::Mat
CVImageDeblurring::sharp_image (const cv::Mat &gray_blurred, float sigmar)
{
    cv::Mat image;
    gray_blurred.convertTo (image, CV_32FC1);
    cv::Mat bilateral_image;
    cv::bilateralFilter (gray_blurred, bilateral_image, 5, sigmar, 2);

    cv::Mat sharpFilter = (cv::Mat_<float>(3, 3) << -1, -1, -1, -1, 8, -1, -1, -1, -1);
    cv::Mat filtered_image;
    cv::filter2D (bilateral_image, filtered_image, -1, sharpFilter);
    filtered_image.convertTo (filtered_image, CV_32FC1);
    double minVal;
    double maxVal;
    cv::minMaxLoc (filtered_image, &minVal, &maxVal);
    filtered_image -= (float)minVal;
    filtered_image *= (255.0f / maxVal);
    cv::Mat sharpened = image + filtered_image;
    cv::minMaxLoc(sharpened, &minVal, &maxVal);
    sharpened *= (255.0 / maxVal);
    return sharpened.clone ();
}


float
CVImageDeblurring::measure_sharp (const cv::Mat &gray_blurred)
{
    cv::Mat dst;
    cv::Laplacian (gray_blurred, dst, -1, 3, 1, 0, cv::BORDER_CONSTANT);
    dst.convertTo (dst, CV_8UC1);
    float sum = 0;
    for (int i = 0; i < gray_blurred.rows; i++)
    {
        for (int j = 0; j < gray_blurred.cols; j++)
        {
            sum += dst.at<unsigned char>(i, j);
        }
    }
    sum /= (gray_blurred.rows * gray_blurred.cols);
    return sum;
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
CVImageDeblurring::rotate (cv::Mat &src, cv::Mat &dst)
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
CVImageDeblurring::blind_deblurring (const cv::Mat &blurred, cv::Mat &deblurred, cv::Mat &kernel)
{
    cv::Mat gray_blurred;
    cv::cvtColor (blurred, gray_blurred, CV_BGR2GRAY);
    cv::Mat median_blurred;
    medianBlur (gray_blurred, median_blurred, 3);
    float noise_power = 1.0f / _helper->get_snr (gray_blurred, median_blurred);
    XCAM_LOG_DEBUG("estimated inv snr %f", noise_power);
    std::vector<cv::Mat> blurred_rgb(3);
    cv::split(blurred, blurred_rgb);
    std::vector<cv::Mat> deblurred_rgb(3);
    int kernel_size = estimate_kernel_size (gray_blurred);
    XCAM_LOG_DEBUG("estimated kernel size %d", kernel_size);
    cv::Mat result_deblurred;
    cv::Mat result_kernel;
    blind_deblurring_one_channel (gray_blurred, result_kernel, kernel_size, noise_power);
    for (int i = 0; i < 3; i++)
    {
        wiener_filter (edgetaper(blurred_rgb[i], result_kernel), result_kernel, deblurred_rgb[i], noise_power);
    }
    cv::merge (deblurred_rgb, result_deblurred);
    deblurred = result_deblurred.clone();
    kernel = result_kernel.clone();
}

void
CVImageDeblurring::blind_deblurring_one_channel (const cv::Mat &blurred, cv::Mat &kernel, int kernel_size, float noise_power)
{
    cv::Mat kernel_current = cv::Mat::zeros (kernel_size, kernel_size, CV_32FC1);
    cv::Mat deblurred_current = _helper->erosion (blurred, 2, 0);
    float sigmar = 20;
    cv::Mat enhanced_blurred = blurred.clone ();
    for (int i = 0; i < _config.iterations; i++)
    {
        cv::Mat sharpened = sharp_image (deblurred_current, sigmar);
        wiener_filter(blurred, sharpened.clone (), kernel_current, noise_power);
        kernel_current = kernel_current (cv::Rect((blurred.cols - kernel_size) / 2 , (blurred.rows - kernel_size) / 2, kernel_size, kernel_size));
        double min_val;
        double max_val;
        cv::minMaxLoc (kernel_current, &min_val, &max_val);
        _helper->apply_constraints (kernel_current, (float)max_val / 15);
        _helper->normalize_weights (kernel_current);
        enhanced_blurred = edgetaper (blurred, kernel_current);
        wiener_filter (enhanced_blurred, kernel_current.clone(), deblurred_current, noise_power);
        _helper->apply_constraints (deblurred_current, 0);
        sigmar *= 0.9;
    }
    kernel = kernel_current.clone ();
}

void
CVImageDeblurring::wiener_filter (const cv::Mat &blurred_image, const cv::Mat &known, cv::Mat &unknown, float noise_power)
{
    int image_w = blurred_image.size().width;
    int image_h = blurred_image.size().height;
    cv::Mat yFT[2];
    _helper->compute_dft (blurred_image, yFT);

    cv::Mat padded = cv::Mat::zeros(image_h, image_w, CV_32FC1);
    int padx = padded.cols - known.cols;
    int pady = padded.rows - known.rows;
    cv::copyMakeBorder (known, padded, pady / 2, pady - pady / 2, padx / 2, padx - padx / 2, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    cv::Mat padded_ft[2];
    _helper->compute_dft (padded, padded_ft);

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
    _helper->compute_idft (unknown_ft, temp_unknown);
    rotate (temp_unknown, temp_unknown);
    unknown = temp_unknown.clone();
}

}
