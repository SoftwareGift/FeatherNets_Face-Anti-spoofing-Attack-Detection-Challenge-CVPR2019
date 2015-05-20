/*
 * cl_tnr_handler.h - CL tnr handler
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
 * Author: Wei Zong <wei.zong@intel.com>
 */

#ifndef XCAM_CL_DEMO_HANLDER_H
#define XCAM_CL_DEMO_HANLDER_H

#include "xcam_utils.h"
#include "cl_image_handler.h"

namespace XCam {

enum CLTnrType {
    CL_TNR_DISABLE = 0,
    CL_TNR_TYPE_YUV = 1 << 0,
    CL_TNR_TYPE_RGB = 1 << 1,
};

#define TNR_PROCESSING_FRAME_COUNT  3

class CLTnrImageKernel
    : public CLImageKernel
{
    typedef std::list<SmartPtr<CLImage>> CLImagePtrList;

public:
    explicit CLTnrImageKernel (SmartPtr<CLContext> &context,
                               const char *name,
                               CLTnrType type);

    virtual ~CLTnrImageKernel () {
        _image_in_list.clear ();
    }

    CLTnrType get_type () {
        return _type;
    }

    uint32_t get_frameCount () {
        return _frame_count;
    }

    bool set_gain (float gain);
    bool set_threshold (float thr_y, float thr_uv);

    virtual XCamReturn post_execute ();
protected:
    virtual XCamReturn prepare_arguments (
        SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
        CLArgument args[], uint32_t &arg_count,
        CLWorkSize &work_size);

private:
    XCAM_DEAD_COPY (CLTnrImageKernel);

    CLTnrType _type;
    float    _gain;
    float    _thr_Y;
    float    _thr_C;

    uint32_t _vertical_offset;
    uint8_t  _frame_count;
    CLImagePtrList _image_in_list;
    SmartPtr<CLImage> _image_out_prev;
};

class CLTnrImageHandler
    : public CLImageHandler
{
public:
    explicit CLTnrImageHandler (const char *name);
    bool set_tnr_kernel (SmartPtr<CLTnrImageKernel> &kernel);
    bool set_mode (uint32_t mode);
    bool set_gain (float gain);
    bool set_threshold (float thr_y, float thr_uv);

private:
    XCAM_DEAD_COPY (CLTnrImageHandler);

private:
    SmartPtr<CLTnrImageKernel>  _tnr_kernel;
    CLTnrType _mode;
};

SmartPtr<CLImageHandler>
create_cl_tnr_image_handler (SmartPtr<CLContext> &context, CLTnrType type);

};

#endif //XCAM_CL_DEMO_HANLDER_H
