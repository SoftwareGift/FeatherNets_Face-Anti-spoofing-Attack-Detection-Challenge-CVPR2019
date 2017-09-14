/*
 * stablizer.h - abstract header for DVS (Digital Video Stabilizer)
 *
 *    Copyright (c) 2014-2016 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Author: Zong Wei <wei.zong@intel.com>
 */

#ifndef _STABILIZER_H_
#define _STABILIZER_H_

#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videostab.hpp>

#include "libdvs.h"

class OnePassVideoStabilizer : public cv::videostab::OnePassStabilizer
{
public:
    virtual ~OnePassVideoStabilizer() {};

    virtual cv::Mat nextStabilizedMotion(DvsData* frame, int& stablizedPos);

protected:
    virtual cv::Mat estimateMotion();
    virtual void setUpFrame(const cv::Mat &firstFrame);

private:

};

class TwoPassVideoStabilizer : public cv::videostab::TwoPassStabilizer
{
public:
    virtual ~TwoPassVideoStabilizer() {};

    virtual cv::Mat nextStabilizedMotion(DvsData* frame, int& stablizedPos);

protected:
    virtual cv::Mat estimateMotion();
    virtual void setUpFrame(const cv::Mat &firstFrame);

private:

};

class VideoStabilizer
{
public:
    VideoStabilizer(bool isTwoPass = false,
                    bool wobbleSuppress = false,
                    bool deblur = false,
                    bool inpainter = false);
    virtual ~VideoStabilizer();

    cv::Ptr<cv::videostab::StabilizerBase> stabilizer() const {
        return stabilizer_;
    }

    cv::Mat nextFrame();
    cv::Mat nextStabilizedMotion(DvsData* frame, int& stablizedPos);

    cv::Size trimedVideoSize(cv::Size frameSize);
    cv::Mat cropVideoFrame(cv::Mat& frame);

    void setFrameSize(cv::Size frameSize);
    cv::Size getFrameSize() const {
        return frameSize_;
    }

    void configFeatureDetector(int features, double minDistance);
    void configMotionFilter(int radius, float stdev);
    void configDeblurrer(int radius, double sensitivity);

public:
    cv::VideoWriter writer_;

private:
    bool isTwoPass_;
    float trimRatio_;
    cv::Size frameSize_;
    cv::Ptr<cv::videostab::StabilizerBase> stabilizer_;
};


#endif
