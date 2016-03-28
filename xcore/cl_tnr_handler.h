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

#ifndef XCAM_CL_TNR_HANLDER_H
#define XCAM_CL_TNR_HANLDER_H

#include "xcam_utils.h"
#include "cl_image_handler.h"
#include "base/xcam_3a_result.h"

namespace XCam {

enum CLTnrType {
    CL_TNR_DISABLE = 0,
    CL_TNR_TYPE_YUV = 1 << 0,
    CL_TNR_TYPE_RGB = 1 << 1,
};

enum CLTnrHistogramType {
    CL_TNR_HIST_BRIGHTNESS   = 0,
    CL_TNR_HIST_HOR_PROJECTION = 1,
    CL_TNR_HIST_VER_PROJECTION = 2,
};

enum CLTnrAnalyzeDateType {
    CL_TNR_ANALYZE_STATS = 0,
    CL_TNR_ANALYZE_RGB   = 1,
};

typedef struct _CLTnrMotionInfo {
    int32_t hor_shift; /*!< pixel count of horizontal direction (X) shift  */
    int32_t ver_shift; /*!< pixel count of vertical direction (Y) shift  */
    float hor_corr;   /*!< horizontal direction (X) correlation */
    float ver_corr;   /*!< vertical direction (Y) correlation */
} CLTnrMotionInfo;

#define TNR_PROCESSING_FRAME_COUNT  4
#define TNR_LIST_FRAME_COUNT  4
#define TNR_GRID_HOR_COUNT          8
#define TNR_GRID_VER_COUNT          8
#define TNR_MOTION_THRESHOLD        2

class CLTnrImageKernel
    : public CLImageKernel
{
    typedef std::list<SmartPtr<CLImage>> CLImagePtrList;
    typedef std::list<CLTnrMotionInfo> CLTnrMotionInfoList;

private:
    struct CLTnrHistogram {
        CLTnrHistogram ();
        CLTnrHistogram (uint32_t width, uint32_t height);
        ~CLTnrHistogram ();

        XCAM_DEAD_COPY (CLTnrHistogram);

        float*   hor_hist_current;
        float*   hor_hist_reference;
        float*   ver_hist_current;
        float*   ver_hist_reference;
        uint32_t hor_hist_bin;
        uint32_t ver_hist_bin;
    };

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

    bool set_rgb_config (const XCam3aResultTemporalNoiseReduction& config);
    bool set_yuv_config (const XCam3aResultTemporalNoiseReduction& config);
    bool set_framecount (uint8_t count) ;

    virtual XCamReturn post_execute (SmartPtr<DrmBoBuffer> &output);
protected:
    virtual XCamReturn prepare_arguments (
        SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
        CLArgument args[], uint32_t &arg_count,
        CLWorkSize &work_size);

private:
    XCAM_DEAD_COPY (CLTnrImageKernel);

    bool calculate_image_histogram (XCam3AStats *stats, CLTnrHistogramType type, float* histogram);
    bool calculate_image_histogram (SmartPtr<DrmBoBuffer> &input, CLTnrHistogramType type, float* histogram);
    void print_image_histogram ();

    CLTnrType _type;
    float    _gain_yuv;
    float    _thr_y;
    float    _thr_uv;
    float    _gain_rgb;
    float    _thr_r;
    float    _thr_g;
    float    _thr_b;
    uint8_t  _frame_count;
    uint8_t  _stable_frame_count;

    CLTnrHistogram _image_histogram;
    CLTnrMotionInfo _motion_info[TNR_GRID_HOR_COUNT * TNR_GRID_VER_COUNT];

    uint32_t _vertical_offset;
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
    bool set_framecount (uint8_t count) ;
    bool set_rgb_config (const XCam3aResultTemporalNoiseReduction& config);
    bool set_yuv_config (const XCam3aResultTemporalNoiseReduction& config);

private:
    XCAM_DEAD_COPY (CLTnrImageHandler);

private:
    SmartPtr<CLTnrImageKernel>  _tnr_kernel;
    CLTnrType _mode;
};

SmartPtr<CLImageHandler>
create_cl_tnr_image_handler (SmartPtr<CLContext> &context, CLTnrType type);

};

#endif //XCAM_CL_TNR_HANLDER_H