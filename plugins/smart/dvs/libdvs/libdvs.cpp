/*
 * libdvs.cpp - abstract for DVS (Digital Video Stabilizer)
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

#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/core/utility.hpp>

#include "libdvs.h"
#include "stabilizer.h"

struct DigitalVideoStabilizer : DvsInterface
{
    virtual ~DigitalVideoStabilizer() {}

    int init(int width, int height, bool twoPass);

    void setConfig(DvsConfig* config);

    void release();

    void nextStabilizedMotion(DvsData* frame, DvsResult* result);


    VideoStabilizer* _videoStab;

    DigitalVideoStabilizer () {
        _videoStab = NULL;
    }
};

int DigitalVideoStabilizer::init(int width, int height, bool twoPass)
{
    Size frameSize;
    frameSize.width = width;
    frameSize.height = height;

    if (_videoStab != NULL) {
        delete _videoStab;
        _videoStab = NULL;
    }
    _videoStab = new VideoStabilizer(twoPass, false, false, false);
    if (_videoStab == NULL) {
        return -1;
    }

    // stabilizer configuration
    _videoStab->setFrameSize(frameSize);
    _videoStab->configFeatureDetector(1000, 15);

    return 0;
}

void DigitalVideoStabilizer::setConfig(DvsConfig* config)
{
    if (NULL == _videoStab) {
        return;
    }
    // stabilizer configuration
    _videoStab->setFrameSize(Size(config->frame_width, config->frame_height));
    _videoStab->configMotionFilter(config->radius, config->stdev);
    _videoStab->configFeatureDetector(config->features, config->minDistance);
}

void DigitalVideoStabilizer::release()
{
    if (_videoStab != NULL) {
        delete _videoStab;
    }
}

void DigitalVideoStabilizer::nextStabilizedMotion(DvsData* frame, DvsResult* result)
{
    if ((NULL == _videoStab) || (NULL == result)) {
        return;
    }
    result->frame_id = -1;
    result->frame_width = _videoStab->getFrameSize().width;
    result->frame_height = _videoStab->getFrameSize().height;

    cv::Mat HMatrix = _videoStab->nextStabilizedMotion(frame, result->frame_id);

    if (HMatrix.empty()) {
        result->valid = false;
        result->proj_mat[0][0] = 1.0f;
        result->proj_mat[0][1] = 0.0f;
        result->proj_mat[0][2] = 0.0f;
        result->proj_mat[1][0] = 0.0f;
        result->proj_mat[1][1] = 1.0f;
        result->proj_mat[1][2] = 0.0f;
        result->proj_mat[2][0] = 0.0f;
        result->proj_mat[2][1] = 0.0f;
        result->proj_mat[2][2] = 1.0f;
        return;
    }

    cv::Mat invHMat = HMatrix.inv();
    result->valid = true;

    for( int i = 0; i < 3; i++ ) {
        for( int j = 0; j < 3; j++ ) {
            result->proj_mat[i][j] = invHMat.at<float>(i, j);
        }
    }
#if 0
    printf ("proj_mat(%d, :)={%f, %f, %f, %f, %f, %f, %f, %f, %f}; \n", result->frame_id + 1,
            result->proj_mat[0][0], result->proj_mat[0][1], result->proj_mat[0][2],
            result->proj_mat[1][0], result->proj_mat[1][1], result->proj_mat[1][2],
            result->proj_mat[2][0], result->proj_mat[2][1], result->proj_mat[2][2]);

    printf ("amplitude(%d, :)={%f, %f}; \n", result->frame_id + 1,
            result->proj_mat[0][2], result->proj_mat[1][2]);
#endif
}

DvsInterface* getDigitalVideoStabilizer(void)
{
    return new DigitalVideoStabilizer();
}



