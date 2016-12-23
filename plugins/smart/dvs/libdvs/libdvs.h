/*
 * libdvs.h - abstract header for DVS (Digital Video Stabilizer)
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

#ifndef _LIB_DVS_HPP
#define _LIB_DVS_HPP

#include <stdio.h>

#if (defined __linux__)
#define DVSAPI __attribute__((visibility("default")))
#endif

typedef struct DvsData
{
    cv::UMat data;

    virtual ~DvsData () {};
} DvsData;


typedef struct DvsResult
{
    int frame_id;
    bool valid;
    int frame_width;
    int frame_height;
    double proj_mat[3][3];

    DvsResult(): frame_id(-1), valid(false)
    {};
} DvsResult;


typedef struct  DvsConfig
{
    bool use_ocl; //ture:ocl path; false:cpu path;
    int frame_width;
    int frame_height;
    int radius;
    float stdev;
    int features;
    double minDistance;

    DvsConfig()
    {
        use_ocl = true;
        frame_width = 1;
        frame_height = 1;
        radius = 15;
        stdev = 10.0f;
        features = 1000;
        minDistance = 15.0f;
    }
} DvsConfig;

typedef struct DvsInterface
{
    virtual ~DvsInterface() {}
    /// initialize model from memory
    virtual int init(int width, int height, bool twoPass) = 0;

    /// set detection parameters, if config = NULL, default parameters will be used
    virtual void setConfig(DvsConfig* config) = 0;

    /// release memory
    virtual void release() = 0;

    /// apply homography estimation to an input image
    /// @param frame input 8-bit single channel UMAT image (color image must be transferred to gray-scale)
    /// @param result output homography estimation result of the input image
    virtual void nextStabilizedMotion(DvsData* frame, DvsResult* result) = 0;

} DvsInterface;

extern "C" DVSAPI DvsInterface* getDigitalVideoStabilizer(void);

#endif


