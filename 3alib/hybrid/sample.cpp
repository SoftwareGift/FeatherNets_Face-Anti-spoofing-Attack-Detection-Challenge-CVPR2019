/*
 * aiq_wrapper.cpp - aiq wrapper:
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
 * Author: Jia Meng <jia.meng@intel.com>
 */

#include <base/xcam_3a_description.h>
#include "xcam_utils.h"

using namespace XCam;

#define CONTEXT_CAST(context)  ((XCam3AHybridContext*)(context))

class XCam3AHybridContext
{
public:
    XCam3AHybridContext ();
    ~XCam3AHybridContext ();

private:
    XCAM_DEAD_COPY (XCam3AHybridContext);

};

XCam3AHybridContext::XCam3AHybridContext ()
{
}

XCam3AHybridContext::~XCam3AHybridContext ()
{
}

static XCamReturn
xcam_create_context (XCam3AContext **context)
{
    XCAM_ASSERT (context);
    XCam3AHybridContext *ctx = new XCam3AHybridContext ();
    *context = ((XCam3AContext*)(ctx));
    return XCAM_RETURN_NO_ERROR;
}

static XCamReturn
xcam_destroy_context (XCam3AContext *context)
{
    XCam3AHybridContext *ctx = CONTEXT_CAST (context);
    delete ctx;
    return XCAM_RETURN_NO_ERROR;
}

// configure customized 3a analyzer with width/height/framerate
static XCamReturn
xcam_configure_3a (XCam3AContext *context, uint32_t width, uint32_t height, double framerate)
{
    XCAM_UNUSED (context);
    XCAM_UNUSED (width);
    XCAM_UNUSED (height);
    XCAM_UNUSED (framerate);

    return XCAM_RETURN_NO_ERROR;
}

// set 3a stats to customized 3a analyzer for subsequent usage
static XCamReturn
xcam_set_3a_stats (XCam3AContext *context, XCam3AStats *stats, int64_t timestamp)
{
    XCAM_UNUSED (context);
    XCAM_UNUSED (timestamp);

    XCam3AStatsInfo info = stats->info;
    for (uint32_t i = 0; i < info.height; ++i)
        for (uint32_t j = 0; j < info.width; ++j) {
            XCAM_LOG_DEBUG ("%d %d %d %d %d %d %d %d",
                            stats->stats[i * info.aligned_width + j].avg_y,
                            stats->stats[i * info.aligned_width + j].avg_gr,
                            stats->stats[i * info.aligned_width + j].avg_r,
                            stats->stats[i * info.aligned_width + j].avg_b,
                            stats->stats[i * info.aligned_width + j].avg_gb,
                            stats->stats[i * info.aligned_width + j].valid_wb_count,
                            stats->stats[i * info.aligned_width + j].f_value1,
                            stats->stats[i * info.aligned_width + j].f_value2);
        }

    return XCAM_RETURN_NO_ERROR;
}

// refer to xcam_params.h for common parameters
static XCamReturn
xcam_update_common_params (XCam3AContext *context, XCamCommonParam *params)
{
    XCAM_UNUSED (context);
    XCAM_UNUSED (params);

    return XCAM_RETURN_NO_ERROR;
}

// customized awb algorithm should be added here
static XCamReturn
xcam_analyze_awb (XCam3AContext *context, XCamAwbParam *params)
{
    XCAM_UNUSED (context);
    XCAM_UNUSED (params);

    return XCAM_RETURN_NO_ERROR;
}

// customized ae algorithm should be added here
static XCamReturn
xcam_analyze_ae (XCam3AContext *context, XCamAeParam *params)
{
    XCAM_UNUSED (context);
    XCAM_UNUSED (params);

    return XCAM_RETURN_NO_ERROR;
}

// customized af is unsupported now
static XCamReturn
xcam_analyze_af (XCam3AContext *context, XCamAfParam *params)
{
    XCAM_UNUSED (context);
    XCAM_UNUSED (params);

    return XCAM_RETURN_NO_ERROR;
}

// combine ae/awb analyze results and set to framework
// only support XCam3aResultExposure and XCam3aResultWhiteBalance now
static XCamReturn
xcam_combine_analyze_results (XCam3AContext *context, XCam3aResultHead *results[], uint32_t *res_count)
{
    XCAM_UNUSED (context);

    uint32_t result_count = 2;
    static XCam3aResultHead *res_array[XCAM_3A_MAX_RESULT_COUNT];
    xcam_mem_clear (res_array);

    for (uint32_t i = 0; i < result_count; ++i) {
        results[i] = res_array[i];
    }
    *res_count = result_count;

    XCam3aResultExposure *exposure = xcam_malloc0_type (XCam3aResultExposure);
    XCAM_ASSERT (exposure);
    exposure->head.type = XCAM_3A_RESULT_EXPOSURE;
    exposure->head.process_type = XCAM_IMAGE_PROCESS_ALWAYS;
    exposure->head.version = XCAM_VERSION;
    exposure->exposure_time = 9986; // 9.986ms
    exposure->analog_gain = 10;
    results[0] = (XCam3aResultHead *)exposure;

    XCam3aResultWhiteBalance *wb = xcam_malloc0_type (XCam3aResultWhiteBalance);
    XCAM_ASSERT (wb);
    wb->head.type = XCAM_3A_RESULT_WHITE_BALANCE;
    wb->head.process_type = XCAM_IMAGE_PROCESS_ALWAYS;
    wb->head.version = XCAM_VERSION;
    wb->gr_gain = 1.0;
    wb->r_gain = 1.6453;
    wb->b_gain = 2.0645;
    wb->gb_gain = 1.0;
    results[1] = (XCam3aResultHead *)wb;

    return XCAM_RETURN_NO_ERROR;
}

static void
xcam_free_results (XCam3aResultHead *results[], uint32_t res_count)
{
    for (uint32_t i = 0; i < res_count; ++i) {
        if (results[i])
            xcam_free (results[i]);
    }
}

XCAM_BEGIN_DECLARE

XCam3ADescription xcam_3a_desciption = {
    XCAM_VERSION,
    sizeof (XCam3ADescription),
    xcam_create_context,
    xcam_destroy_context,
    xcam_configure_3a,
    xcam_set_3a_stats,
    xcam_update_common_params,
    xcam_analyze_awb,
    xcam_analyze_ae,
    xcam_analyze_af,
    xcam_combine_analyze_results,
    xcam_free_results
};

XCAM_END_DECLARE

