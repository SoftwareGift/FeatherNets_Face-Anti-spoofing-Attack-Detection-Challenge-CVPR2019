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

#include "cl_utils.h"
#include "base/xcam_3a_result.h"
#include "x3a_stats_pool.h"
#include "ocl/cl_image_handler.h"

namespace XCam {

enum CLTnrType {
    CL_TNR_DISABLE = 0,
    CL_TNR_TYPE_YUV = 1 << 0,
    CL_TNR_TYPE_RGB = 1 << 1,
};

#define TNR_GRID_HOR_COUNT          8
#define TNR_GRID_VER_COUNT          8

class CLTnrImageKernel
    : public CLImageKernel
{
public:
    explicit CLTnrImageKernel (
        const SmartPtr<CLContext> &context, CLTnrType type);

    virtual ~CLTnrImageKernel () {
    }

private:
    CLTnrType          _type;
};

class CLTnrImageHandler
    : public CLImageHandler
{
private:
    typedef std::list<SmartPtr<CLImage>> CLImagePtrList;

    enum CLTnrHistogramType {
        CL_TNR_HIST_BRIGHTNESS   = 0,
        CL_TNR_HIST_HOR_PROJECTION = 1,
        CL_TNR_HIST_VER_PROJECTION = 2,
    };

    enum CLTnrAnalyzeDateType {
        CL_TNR_ANALYZE_STATS = 0,
        CL_TNR_ANALYZE_RGB   = 1,
    };

    struct CLTnrMotionInfo {
        int32_t hor_shift; /*!< pixel count of horizontal direction (X) shift  */
        int32_t ver_shift; /*!< pixel count of vertical direction (Y) shift  */
        float hor_corr;   /*!< horizontal direction (X) correlation */
        float ver_corr;   /*!< vertical direction (Y) correlation */
        CLTnrMotionInfo ();
    };

    typedef std::list<CLTnrMotionInfo> CLTnrMotionInfoList;

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
    explicit CLTnrImageHandler (const SmartPtr<CLContext> &context, CLTnrType type, const char *name);
    bool set_tnr_kernel (SmartPtr<CLTnrImageKernel> &kernel);
    bool set_framecount (uint8_t count) ;
    bool set_rgb_config (const XCam3aResultTemporalNoiseReduction& config);
    bool set_yuv_config (const XCam3aResultTemporalNoiseReduction& config);
    uint32_t get_frame_count () {
        return _frame_count;
    }

protected:
    virtual XCamReturn prepare_parameters (SmartPtr<VideoBuffer> &input, SmartPtr<VideoBuffer> &output);

private:
    XCAM_DEAD_COPY (CLTnrImageHandler);

    bool calculate_image_histogram (XCam3AStats *stats, CLTnrHistogramType type, float* histogram);
    bool calculate_image_histogram (SmartPtr<VideoBuffer> &input, CLTnrHistogramType type, float* histogram);
    void print_image_histogram ();

private:
    SmartPtr<CLTnrImageKernel>  _tnr_kernel;
    CLTnrType                   _type;

    float                       _gain_yuv;
    float                       _thr_y;
    float                       _thr_uv;

    float                       _gain_rgb;
    float                       _thr_r;
    float                       _thr_g;
    float                       _thr_b;

    CLTnrMotionInfo             _motion_info[TNR_GRID_HOR_COUNT * TNR_GRID_VER_COUNT];
    CLImagePtrList              _image_in_list;
    CLTnrHistogram              _image_histogram;
    SmartPtr<CLImage>           _image_out_prev;

    uint8_t                     _frame_count;
};

SmartPtr<CLImageHandler>
create_cl_tnr_image_handler (const SmartPtr<CLContext> &context, CLTnrType type);

};

#endif //XCAM_CL_TNR_HANLDER_H
