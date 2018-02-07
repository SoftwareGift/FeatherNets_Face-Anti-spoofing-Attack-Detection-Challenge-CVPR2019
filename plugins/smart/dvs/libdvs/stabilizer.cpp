/*
 * stablizer.cpp - abstract for DVS (Digital Video Stabilizer)
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

#include "stabilizer.h"

using namespace cv;
using namespace cv::videostab;
using namespace std;

Mat
OnePassVideoStabilizer::nextStabilizedMotion(DvsData* frame, int& stablizedPos)
{
    if (!(frame->data.empty()))
    {
        Mat image;
        frame->data.getMat(ACCESS_READ).copyTo(image);

        curPos_++;

        if (curPos_ > 0)
        {
            at(curPos_, frames_) = image;

            if (doDeblurring_)
                at(curPos_, blurrinessRates_) = calcBlurriness(image);

            at(curPos_ - 1, motions_) = estimateMotion();

            if (curPos_ >= radius_)
            {
                curStabilizedPos_ = curPos_ - radius_;
                Mat stabilizationMotion = estimateStabilizationMotion();
                if (doCorrectionForInclusion_)
                    stabilizationMotion = ensureInclusionConstraint(stabilizationMotion, frameSize_, trimRatio_);

                at(curStabilizedPos_, stabilizationMotions_) = stabilizationMotion;
                stablizedPos = curStabilizedPos_;

                return stabilizationMotion;
            }
        }
        else
            setUpFrame(image);

        log_->print(".");
        return Mat();
    }

    if (curStabilizedPos_ < curPos_)
    {
        curStabilizedPos_++;
        stablizedPos = curStabilizedPos_;
        at(curStabilizedPos_ + radius_, frames_) = at(curPos_, frames_);
        at(curStabilizedPos_ + radius_ - 1, motions_) = Mat::eye(3, 3, CV_32F);

        Mat stabilizationMotion = estimateStabilizationMotion();
        if (doCorrectionForInclusion_)
            stabilizationMotion = ensureInclusionConstraint(stabilizationMotion, frameSize_, trimRatio_);

        at(curStabilizedPos_, stabilizationMotions_) = stabilizationMotion;

        log_->print(".");

        return stabilizationMotion;
    }

    return Mat();
}


Mat
OnePassVideoStabilizer::estimateMotion()
{
#if ENABLE_DVS_CL_PATH
    cv::UMat frame0 = at(curPos_ - 1, frames_).getUMat(ACCESS_READ);
    cv::UMat frame1 = at(curPos_, frames_).getUMat(ACCESS_READ);

    cv::UMat ugrayImage0;
    cv::UMat ugrayImage1;
    if ( frame0.type() != CV_8U )
    {
        cvtColor( frame0, ugrayImage0, COLOR_BGR2GRAY );
    }
    else
    {
        ugrayImage0 = frame0;
    }

    if ( frame1.type() != CV_8U )
    {
        cvtColor( frame1, ugrayImage1, COLOR_BGR2GRAY );
    }
    else
    {
        ugrayImage1 = frame1;
    }

    return motionEstimator_.dynamicCast<KeypointBasedMotionEstimator>()->estimate(ugrayImage0, ugrayImage1);
#else
    return motionEstimator_.dynamicCast<KeypointBasedMotionEstimator>()->estimate(at(curPos_ - 1, frames_), at(curPos_, frames_));
#endif
}

void
OnePassVideoStabilizer::setUpFrame(const Mat &firstFrame)
{
    frameSize_ = firstFrame.size();
    frameMask_.create(frameSize_, CV_8U);
    frameMask_.setTo(255);

    int cacheSize = 2 * radius_ + 1;
    frames_.resize(2);
    motions_.resize(cacheSize);
    stabilizationMotions_.resize(cacheSize);

    for (int i = -radius_; i < 0; ++i)
    {
        at(i, motions_) = Mat::eye(3, 3, CV_32F);
        at(i, frames_) = firstFrame;
    }

    at(0, frames_) = firstFrame;

    StabilizerBase::setUp(firstFrame);
}


Mat
TwoPassVideoStabilizer::nextStabilizedMotion(DvsData* frame, int& stablizedPos)
{
    if (!(frame->data.empty()))
    {
        Mat image;
        frame->data.getMat(ACCESS_READ).copyTo(image);

        curPos_++;

        if (curPos_ > 0)
        {
            at(curPos_, frames_) = image;

            if (doDeblurring_)
                at(curPos_, blurrinessRates_) = calcBlurriness(image);

            at(curPos_ - 1, motions_) = estimateMotion();

            if (curPos_ >= radius_)
            {
                curStabilizedPos_ = curPos_ - radius_;
                Mat stabilizationMotion = estimateStabilizationMotion();
                if (doCorrectionForInclusion_)
                    stabilizationMotion = ensureInclusionConstraint(stabilizationMotion, frameSize_, trimRatio_);

                at(curStabilizedPos_, stabilizationMotions_) = stabilizationMotion;
                stablizedPos = curStabilizedPos_;

                return stabilizationMotion;
            }
        }
        else
            setUpFrame(image);

        log_->print(".");
        return Mat();
    }

    if (curStabilizedPos_ < curPos_)
    {
        curStabilizedPos_++;
        stablizedPos = curStabilizedPos_;
        at(curStabilizedPos_ + radius_, frames_) = at(curPos_, frames_);
        at(curStabilizedPos_ + radius_ - 1, motions_) = Mat::eye(3, 3, CV_32F);

        Mat stabilizationMotion = estimateStabilizationMotion();
        if (doCorrectionForInclusion_)
            stabilizationMotion = ensureInclusionConstraint(stabilizationMotion, frameSize_, trimRatio_);

        at(curStabilizedPos_, stabilizationMotions_) = stabilizationMotion;

        log_->print(".");

        return stabilizationMotion;
    }

    return Mat();
}


Mat
TwoPassVideoStabilizer::estimateMotion()
{
#if ENABLE_DVS_CL_PATH
    cv::UMat frame0 = at(curPos_ - 1, frames_).getUMat(ACCESS_READ);
    cv::UMat frame1 = at(curPos_, frames_).getUMat(ACCESS_READ);

    cv::UMat ugrayImage0;
    cv::UMat ugrayImage1;
    if ( frame0.type() != CV_8U )
    {
        cvtColor( frame0, ugrayImage0, COLOR_BGR2GRAY );
    }
    else
    {
        ugrayImage0 = frame0;
    }

    if ( frame1.type() != CV_8U )
    {
        cvtColor( frame1, ugrayImage1, COLOR_BGR2GRAY );
    }
    else
    {
        ugrayImage1 = frame1;
    }

    return motionEstimator_.dynamicCast<KeypointBasedMotionEstimator>()->estimate(ugrayImage0, ugrayImage1);
#else
    return motionEstimator_.dynamicCast<KeypointBasedMotionEstimator>()->estimate(at(curPos_ - 1, frames_), at(curPos_, frames_));
#endif
}

void
TwoPassVideoStabilizer::setUpFrame(const Mat &firstFrame)
{
    //int cacheSize = 2*radius_ + 1;
    frames_.resize(2);
    stabilizedFrames_.resize(2);
    stabilizedMasks_.resize(2);

    for (int i = -1; i <= 0; ++i)
        at(i, frames_) = firstFrame;

    StabilizerBase::setUp(firstFrame);
}

VideoStabilizer::VideoStabilizer(
    bool isTwoPass,
    bool wobbleSuppress,
    bool deblur,
    bool inpainter)
    : isTwoPass_ (isTwoPass)
    , trimRatio_ (0.05f)
{
    Ptr<MotionEstimatorRansacL2> est = makePtr<MotionEstimatorRansacL2>(MM_HOMOGRAPHY);
    Ptr<IOutlierRejector> outlierRejector = makePtr<TranslationBasedLocalOutlierRejector>();
    Ptr<KeypointBasedMotionEstimator> kbest = makePtr<KeypointBasedMotionEstimator>(est);
    kbest->setDetector(GFTTDetector::create(1000, 0.01, 15));
    kbest->setOutlierRejector(outlierRejector);

    if (isTwoPass)
    {
        Ptr<TwoPassVideoStabilizer> twoPassStabilizer = makePtr<TwoPassVideoStabilizer>();
        stabilizer_ = twoPassStabilizer;
        twoPassStabilizer->setEstimateTrimRatio(false);
        twoPassStabilizer->setMotionStabilizer(makePtr<GaussianMotionFilter>(15, 10));

        if (wobbleSuppress) {
            Ptr<MoreAccurateMotionWobbleSuppressorBase> ws = makePtr<MoreAccurateMotionWobbleSuppressor>();

            ws->setMotionEstimator(kbest);
            ws->setPeriod(30);
            twoPassStabilizer->setWobbleSuppressor(ws);
        }
    } else {
        Ptr<OnePassVideoStabilizer> onePassStabilizer = makePtr<OnePassVideoStabilizer>();
        stabilizer_ = onePassStabilizer;
        onePassStabilizer->setMotionFilter(makePtr<GaussianMotionFilter>(15, 10));
    }

    stabilizer_->setMotionEstimator(kbest);

    stabilizer_->setRadius(15);

    if (deblur)
    {
        Ptr<WeightingDeblurer> deblurrer = makePtr<WeightingDeblurer>();
        deblurrer->setRadius(3);
        deblurrer->setSensitivity(0.001f);
        stabilizer_->setDeblurer(deblurrer);
    }

    if (inpainter)
    {
        bool inpaintMosaic = true;
        bool inpaintColorAverage = true;
        bool inpaintColorNs = false;
        bool inpaintColorTelea = false;

        // init inpainter
        InpaintingPipeline *inpainters = new InpaintingPipeline();
        Ptr<InpainterBase> inpainters_(inpainters);
        if (true == inpaintMosaic)
        {
            Ptr<ConsistentMosaicInpainter> inp = makePtr<ConsistentMosaicInpainter>();
            inp->setStdevThresh(10.0f);
            inpainters->pushBack(inp);
        }
        if (true == inpaintColorAverage)
            inpainters->pushBack(makePtr<ColorAverageInpainter>());
        else if (true == inpaintColorNs)
            inpainters->pushBack(makePtr<ColorInpainter>(0, 2));
        else if (true == inpaintColorTelea)
            inpainters->pushBack(makePtr<ColorInpainter>(1, 2));
        if (!inpainters->empty())
        {
            inpainters->setRadius(2);
            stabilizer_->setInpainter(inpainters_);
        }
    }
}

VideoStabilizer::~VideoStabilizer() {}

void
VideoStabilizer::configFeatureDetector(int features, double minDistance)
{
    Ptr<ImageMotionEstimatorBase> estimator = stabilizer_->motionEstimator();
    Ptr<FeatureDetector> detector = estimator.dynamicCast<KeypointBasedMotionEstimator>()->detector();
    if (NULL == detector) {
        return;
    }

    detector.dynamicCast<GFTTDetector>()->setMaxFeatures(features);
    detector.dynamicCast<GFTTDetector>()->setMinDistance(minDistance);
}

void
VideoStabilizer::configMotionFilter(int radius, float stdev)
{
    if (isTwoPass_) {
        Ptr<TwoPassVideoStabilizer> stab = stabilizer_.dynamicCast<TwoPassVideoStabilizer>();
        Ptr<IMotionStabilizer> motionStabilizer = stab->motionStabilizer();
        motionStabilizer.dynamicCast<GaussianMotionFilter>()->setParams(radius, stdev);
    } else {
        Ptr<OnePassVideoStabilizer> stab = stabilizer_.dynamicCast<OnePassVideoStabilizer>();
        Ptr<MotionFilterBase> motionFilter = stab->motionFilter();
        motionFilter.dynamicCast<GaussianMotionFilter>()->setParams(radius, stdev);
    }
    stabilizer_->setRadius(radius);
}

void
VideoStabilizer::configDeblurrer(int radius, double sensitivity)
{
    Ptr<DeblurerBase> deblurrer = stabilizer_->deblurrer();
    if (NULL == deblurrer) {
        return;
    }

    deblurrer->setRadius(radius);
    deblurrer.dynamicCast<WeightingDeblurer>()->setSensitivity(sensitivity);
}

void
VideoStabilizer::setFrameSize(Size frameSize)
{
    frameSize_ = frameSize;
}

Mat
VideoStabilizer::nextFrame()
{
    Mat frame;

    if (isTwoPass_) {
        Ptr<TwoPassVideoStabilizer> stab = stabilizer_.dynamicCast<TwoPassVideoStabilizer>();
        if(!stab.empty())
            frame = stab->nextFrame();
        else
            CV_Error (CV_StsNullPtr, "VideoStabilizer: cast stabilizer failed");
    } else {
        Ptr<OnePassVideoStabilizer> stab = stabilizer_.dynamicCast<OnePassVideoStabilizer>();
        if(!stab.empty())
            frame = stab->nextFrame();
        else
            CV_Error (CV_StsNullPtr, "VideoStabilizer: cast stabilizer failed");
    }

    return frame;
}

Mat
VideoStabilizer::nextStabilizedMotion(DvsData* frame, int& stablizedPos)
{
    Mat HMatrix;

    if (isTwoPass_) {
        Ptr<TwoPassVideoStabilizer> stab = stabilizer_.dynamicCast<TwoPassVideoStabilizer>();
        if(!stab.empty())
            HMatrix = stab->nextStabilizedMotion(frame, stablizedPos);
        else
            CV_Error (CV_StsNullPtr, "VideoStabilizer: cast stabilizer failed");
    } else {
        Ptr<OnePassVideoStabilizer> stab = stabilizer_.dynamicCast<OnePassVideoStabilizer>();
        if(!stab.empty())
            HMatrix = stab->nextStabilizedMotion(frame, stablizedPos);
        else
            CV_Error (CV_StsNullPtr, "VideoStabilizer: cast stabilizer failed");
    }

    return HMatrix;
}

Size
VideoStabilizer::trimedVideoSize(Size frameSize)
{
    Size outputFrameSize;
    outputFrameSize.width = ((int)((float)frameSize.width * (1 - 2 * trimRatio_)) >> 3) << 3;
    outputFrameSize.height = ((int)((float)frameSize.height * (1 - 2 * trimRatio_)) >> 3) << 3;

    return (outputFrameSize);
}

Mat
VideoStabilizer::cropVideoFrame(Mat& frame)
{
    Rect cropROI;
    Size inputFrameSize = frame.size();
    Size outputFrameSize = trimedVideoSize(inputFrameSize);

    cropROI.x = (inputFrameSize.width - outputFrameSize.width) >> 1;
    cropROI.y = (inputFrameSize.height - outputFrameSize.height) >> 1;
    cropROI.width = outputFrameSize.width;
    cropROI.height = outputFrameSize.height;

    Mat croppedFrame = frame(cropROI).clone();

    return croppedFrame;
}

