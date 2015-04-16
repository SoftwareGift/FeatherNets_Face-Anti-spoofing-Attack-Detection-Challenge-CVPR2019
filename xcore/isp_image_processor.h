/*
 * isp_image_processor.h - isp image processor
 *
 *  Copyright (c) 2014-2015 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Author: Wind Yuan <feng.yuan@intel.com>
 */

#ifndef XCAM_ISP_IMAGE_PROCESSOR_H
#define XCAM_ISP_IMAGE_PROCESSOR_H

#include "xcam_utils.h"
#include "image_processor.h"

namespace XCam {

class X3aIspConfig;
class IspController;
class IspConfigTranslator;
class SensorDescriptor;

class IspImageProcessor
    : public ImageProcessor
{
public:
    explicit IspImageProcessor (SmartPtr<IspController> &controller);
    virtual ~IspImageProcessor ();

protected:
    //derive from ImageProcessor
    virtual bool can_process_result (SmartPtr<X3aResult> &result);
    virtual XCamReturn apply_3a_results (X3aResultList &results);
    virtual XCamReturn apply_3a_result (SmartPtr<X3aResult> &result);
    virtual XCamReturn process_buffer (SmartPtr<VideoBuffer> &input, SmartPtr<VideoBuffer> &output);

private:
    XCamReturn merge_results (X3aResultList &results);
    XCamReturn apply_exposure_result (X3aResultList &results);

    XCAM_DEAD_COPY (IspImageProcessor);

private:
    SmartPtr<IspController>          _controller;
    SmartPtr<SensorDescriptor>       _sensor;
    SmartPtr<IspConfigTranslator>    _translator;
    SmartPtr<X3aIspConfig>           _3a_config;
};

class IspExposureImageProcessor
    : public IspImageProcessor
{
public:
    explicit IspExposureImageProcessor (SmartPtr<IspController> &controller);

protected:
    virtual bool can_process_result (SmartPtr<X3aResult> &result);

private:
    XCAM_DEAD_COPY (IspExposureImageProcessor);
};

};

#endif //XCAM_ISP_IMAGE_PROCESSOR_H
