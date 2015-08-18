/*
 * x3a_result.h - 3A calculation result
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

#ifndef XCAM_3A_RESULT_H
#define XCAM_3A_RESULT_H

#include "xcam_utils.h"
#include "smartptr.h"
#include <base/xcam_3a_result.h>
#include <list>

namespace XCam {

class X3aResult
{
protected:
    explicit X3aResult (
        uint32_t type,
        XCamImageProcessType process_type = XCAM_IMAGE_PROCESS_ALWAYS,
        uint64_t timestamp = XCam::InvalidTimestamp
    )
        : _type (type)
        , _process_type (process_type)
        , _timestamp (timestamp)
        , _ptr (NULL)
        , _processed (false)
    {}

public:
    virtual ~X3aResult() {}

    void *get_ptr () const {
        return _ptr;
    }
    bool is_done() const {
        return _processed;
    }
    void set_done (bool flag) {
        _processed = flag;
    }
    uint64_t get_timestamp () const {
        return _timestamp;
    }
    uint32_t get_type () const {
        return _type;
    }

    void set_process_type (XCamImageProcessType process) {
        _process_type = process;
    }
    XCamImageProcessType get_process_type () const {
        return _process_type;
    }

protected:
    void set_ptr (void *ptr) {
        _ptr = ptr;
    }

    //virtual bool to_isp_config (SmartPtr<X3aIspConfig>  &config) = 0;

private:
    XCAM_DEAD_COPY (X3aResult);

protected:
    //XCam3aResultType      _type;
    uint32_t              _type;  // XCam3aResultType
    XCamImageProcessType  _process_type;
    uint64_t              _timestamp;
    void                 *_ptr;
    bool                  _processed;
};

typedef std::list<SmartPtr<X3aResult>>  X3aResultList;

/* !
 * \template StandardResult must inherited from XCam3aResultHead
 */
template <typename StandardResult>
class X3aStandardResultT
    : public X3aResult
{
public:
    explicit X3aStandardResultT (uint32_t type, XCamImageProcessType process_type = XCAM_IMAGE_PROCESS_ALWAYS)
        : X3aResult (type, process_type)
    {
        set_ptr((void*)&_result);
        _result.head.type = (XCam3aResultType)type;
        _result.head.process_type = _process_type;
        _result.head.version = XCAM_VERSION;
    }
    ~X3aStandardResultT () {}

    void set_standard_result (StandardResult &res) {
        uint32_t offset = sizeof (XCam3aResultHead);
        XCAM_ASSERT (sizeof (StandardResult) >= offset);
        memcpy ((uint8_t*)(&_result) + offset, (uint8_t*)(&res) + offset, sizeof (StandardResult) - offset);
    }

    StandardResult &get_standard_result () {
        return _result;
    }
    const StandardResult &get_standard_result () const {
        return _result;
    }

private:
    StandardResult _result;
};

typedef X3aStandardResultT<XCam3aResultWhiteBalance>   X3aWhiteBalanceResult;
typedef X3aStandardResultT<XCam3aResultBlackLevel>     X3aBlackLevelResult;
typedef X3aStandardResultT<XCam3aResultColorMatrix>    X3aColorMatrixResult;
typedef X3aStandardResultT<XCam3aResultExposure>       X3aExposureResult;
typedef X3aStandardResultT<XCam3aResultFocus>          X3aFocusResult;
typedef X3aStandardResultT<XCam3aResultDemosaic>       X3aDemosaicResult;
typedef X3aStandardResultT<XCam3aResultDefectPixel>    X3aDefectPixelResult;
typedef X3aStandardResultT<XCam3aResultNoiseReduction> X3aNoiseReductionResult;
typedef X3aStandardResultT<XCam3aResultEdgeEnhancement>  X3aEdgeEnhancementResult;
typedef X3aStandardResultT<XCam3aResultGammaTable>     X3aGammaTableResult;
typedef X3aStandardResultT<XCam3aResultMaccMatrix>     X3aMaccMatrixResult;
typedef X3aStandardResultT<XCam3aResultChromaToneControl> X3aChromaToneControlResult;
typedef X3aStandardResultT<XCam3aResultBayerNoiseReduction> X3aBayerNoiseReduction;
typedef X3aStandardResultT<XCam3aResultBrightness>      X3aBrightnessResult;
typedef X3aStandardResultT<XCam3aResultTemporalNoiseReduction> X3aTemporalNoiseReduction;
};

#endif //XCAM_3A_RESULT_H
