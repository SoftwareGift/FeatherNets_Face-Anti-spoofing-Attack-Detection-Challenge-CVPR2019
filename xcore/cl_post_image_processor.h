/*
 * cl_post_image_processor.h - CL post image processor
 *
 *  Copyright (c) 2015 Intel Corporation
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
 * Author: Yinhang Liu <yinhangx.liu@intel.com>
 */

#ifndef XCAM_CL_POST_IMAGE_PROCESSOR_H
#define XCAM_CL_POST_IMAGE_PROCESSOR_H

#include "xcam_utils.h"
#include <base/xcam_3a_types.h>
#include "cl_image_processor.h"

namespace XCam {

class CLTnrImageHandler;
class CLRetinexImageHandler;
class CLCscImageHandler;

class CLPostImageProcessor
    : public CLImageProcessor
{
public:
    enum OutSampleType {
        OutSampleYuv,
        OutSampleRGB,
        OutSampleBayer,
    };

    enum CLTnrMode {
        TnrDisable = 0,
        TnrYuv,
    };

public:
    explicit CLPostImageProcessor ();
    virtual ~CLPostImageProcessor ();

    bool set_output_format (uint32_t fourcc);

    virtual bool set_tnr (CLTnrMode mode);
    virtual bool set_retinex (bool enable);

protected:
    virtual bool can_process_result (SmartPtr<X3aResult> &result);
    virtual XCamReturn apply_3a_results (X3aResultList &results);
    virtual XCamReturn apply_3a_result (SmartPtr<X3aResult> &result);

private:
    virtual XCamReturn create_handlers ();

    XCAM_DEAD_COPY (CLPostImageProcessor);

private:
    uint32_t                               _output_fourcc;
    OutSampleType                          _out_sample_type;

    SmartPtr<CLTnrImageHandler>            _tnr;
    SmartPtr<CLRetinexImageHandler>        _retinex;
    SmartPtr<CLCscImageHandler>            _csc;

    CLTnrMode                              _tnr_mode;
    bool                                   _enable_retinex;
};

};
#endif // XCAM_CL_POST_IMAGE_PROCESSOR_H
