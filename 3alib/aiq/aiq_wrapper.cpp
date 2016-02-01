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
 * Author: Wind Yuan <feng.yuan@intel.com>
 */

#include <base/xcam_3a_description.h>
#include "xcam_utils.h"
#include "x3a_analyzer_aiq.h"
#include "x3a_statistics_queue.h"
#include "aiq3a_utils.h"
#include "x3a_result_factory.h"
#include "x3a_analyze_tuner.h"

#define DEFAULT_AIQ_CPF_FILE       "/etc/atomisp/imx185.cpf"


using namespace XCam;

#define AIQ_CONTEXT_CAST(context)  ((XCam3AAiqContext*)(context))

class XCam3AAiqContext
    : public AnalyzerCallback
{
public:
    XCam3AAiqContext ();
    ~XCam3AAiqContext ();
    bool setup_analyzer (struct atomisp_sensor_mode_data &sensor_mode_data, const char *cpf);
    void set_size (uint32_t width, uint32_t height);
    bool setup_stats_pool (uint32_t bit_depth = 8);
    bool is_stats_pool_ready () const {
        return (_stats_pool.ptr () ? true : false);
    }
    SmartPtr<X3aAnalyzeTuner> &get_analyzer () {
        return _analyzer;
    }

    SmartPtr<X3aIspStatistics> get_stats_buffer ();
    uint32_t get_results (X3aResultList &results);

    // derive from AnalyzerCallback
    virtual void x3a_calculation_done (XAnalyzer *analyzer, X3aResultList &results);
    void update_brightness_result(XCamCommonParam *params);

private:
    XCAM_DEAD_COPY (XCam3AAiqContext);

private:
// members
    SmartPtr<X3aAnalyzeTuner>      _analyzer;
    SmartPtr<X3aStatisticsQueue>   _stats_pool;
    uint32_t                       _video_width;
    uint32_t                       _video_height;

    Mutex                          _result_mutex;
    X3aResultList                  _results;
    double                         _brightness_level;
};

XCam3AAiqContext::XCam3AAiqContext ()
    : _video_width (0)
    , _video_height (0)
    , _brightness_level(0)
{
}

XCam3AAiqContext::~XCam3AAiqContext ()
{
    _analyzer->stop ();
    _analyzer->deinit ();
}

bool
XCam3AAiqContext::setup_analyzer (struct atomisp_sensor_mode_data &sensor_mode_data, const char *cpf)
{
    XCAM_ASSERT (!_analyzer.ptr ());
    SmartPtr<X3aAnalyzer> aiq_analyzer = new X3aAnalyzerAiq (sensor_mode_data, cpf);
    XCAM_ASSERT (aiq_analyzer.ptr ());

    _analyzer = new X3aAnalyzeTuner ();
    XCAM_ASSERT (_analyzer.ptr ());

    _analyzer->set_analyzer (aiq_analyzer);
    _analyzer->set_results_callback (this);
    return true;
}

void
XCam3AAiqContext::set_size (uint32_t width, uint32_t height)
{
    _video_width = width;
    _video_height = height;
}

bool
XCam3AAiqContext::setup_stats_pool (uint32_t bit_depth)
{
    VideoBufferInfo info;
    info.init (XCAM_PIX_FMT_SGRBG16, _video_width, _video_height);

    _stats_pool = new X3aStatisticsQueue;
    XCAM_ASSERT (_stats_pool.ptr ());

    _stats_pool->set_bit_depth (bit_depth);
    XCAM_FAIL_RETURN (
        WARNING,
        _stats_pool->set_video_info (info),
        false,
        "3a stats set video info failed");


    if (!_stats_pool->reserve (6)) {
        XCAM_LOG_WARNING ("init_3a_stats_pool failed to reserve stats buffer.");
        return false;
    }

    return true;
}

SmartPtr<X3aIspStatistics>
XCam3AAiqContext::get_stats_buffer ()
{
    SmartPtr<X3aIspStatistics> new_stats =
        _stats_pool->get_buffer (_stats_pool).dynamic_cast_ptr<X3aIspStatistics> ();

    XCAM_FAIL_RETURN (
        WARNING,
        new_stats.ptr (),
        NULL,
        "get isp stats buffer failed");

    return new_stats;
}


void
XCam3AAiqContext::x3a_calculation_done (XAnalyzer *analyzer, X3aResultList &results)
{
    XCAM_UNUSED (analyzer);
    SmartLock  locker (_result_mutex);
    _results.insert (_results.end (), results.begin (), results.end ());
}

void
XCam3AAiqContext::update_brightness_result(XCamCommonParam *params)
{
    if( params->brightness == _brightness_level)
        return;
    _brightness_level = params->brightness;

    XCam3aResultBrightness xcam3a_brightness_result;
    xcam_mem_clear (xcam3a_brightness_result);
    xcam3a_brightness_result.head.type =   XCAM_3A_RESULT_BRIGHTNESS;
    xcam3a_brightness_result.head.process_type = XCAM_IMAGE_PROCESS_ALWAYS;
    xcam3a_brightness_result.head.version = XCAM_VERSION;
    xcam3a_brightness_result.brightness_level = _brightness_level;

    SmartPtr<X3aResult> brightness_result =
        X3aResultFactory::instance ()->create_3a_result ((XCam3aResultHead*)&xcam3a_brightness_result);
    _results.push_back(brightness_result);
}

uint32_t
XCam3AAiqContext::get_results (X3aResultList &results)
{
    uint32_t size = 0;

    SmartLock  locker (_result_mutex);

    results.assign (_results.begin (), _results.end ());
    size = _results.size ();
    _results.clear ();

    return size;
}

static SmartPtr<X3aAnalyzeTuner>
get_analyzer (XCam3AContext *context)
{
    XCam3AAiqContext *aiq_context = AIQ_CONTEXT_CAST (context);
    if (!aiq_context)
        return NULL;

    return aiq_context->get_analyzer ();
}

static XCamReturn
xcam_create_context (XCam3AContext **context)
{
    XCAM_ASSERT (context);
    XCam3AAiqContext *aiq_context = new XCam3AAiqContext ();
    *context = ((XCam3AContext*)(aiq_context));
    return XCAM_RETURN_NO_ERROR;
}

static XCamReturn
xcam_destroy_context (XCam3AContext *context)
{
    XCam3AAiqContext *aiq_context = AIQ_CONTEXT_CAST (context);
    delete aiq_context;
    return XCAM_RETURN_NO_ERROR;
}

static XCamReturn
xcam_configure_3a (XCam3AContext *context, uint32_t width, uint32_t height, double framerate)
{
    XCam3AAiqContext *aiq_context = AIQ_CONTEXT_CAST (context);
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    struct atomisp_sensor_mode_data sensor_mode_data;

    switch ((int)framerate) {
    case 30:
        sensor_mode_data.coarse_integration_time_min = 1;
        sensor_mode_data.coarse_integration_time_max_margin = 1;
        sensor_mode_data.fine_integration_time_min = 0;
        sensor_mode_data.fine_integration_time_max_margin = 0;
        sensor_mode_data.fine_integration_time_def = 0;
        sensor_mode_data.frame_length_lines = 1125;
        sensor_mode_data.line_length_pck = 1100;
        sensor_mode_data.read_mode = 0;
        sensor_mode_data.vt_pix_clk_freq_mhz = 37125000;
        sensor_mode_data.crop_horizontal_start = 0;
        sensor_mode_data.crop_vertical_start = 0;
        sensor_mode_data.crop_horizontal_end = 1920;
        sensor_mode_data.crop_vertical_end = 1080;
        sensor_mode_data.output_width = 1920;
        sensor_mode_data.output_height = 1080;
        sensor_mode_data.binning_factor_x = 1;
        sensor_mode_data.binning_factor_y = 1;
        break;
    default:
        sensor_mode_data.coarse_integration_time_min = 1;
        sensor_mode_data.coarse_integration_time_max_margin = 1;
        sensor_mode_data.fine_integration_time_min = 0;
        sensor_mode_data.fine_integration_time_max_margin = 0;
        sensor_mode_data.fine_integration_time_def = 0;
        sensor_mode_data.frame_length_lines = 1125;
        sensor_mode_data.line_length_pck = 1320;
        sensor_mode_data.read_mode = 0;
        sensor_mode_data.vt_pix_clk_freq_mhz = 37125000;
        sensor_mode_data.crop_horizontal_start = 0;
        sensor_mode_data.crop_vertical_start = 0;
        sensor_mode_data.crop_horizontal_end = 1920;
        sensor_mode_data.crop_vertical_end = 1080;
        sensor_mode_data.output_width = 1920;
        sensor_mode_data.output_height = 1080;
        sensor_mode_data.binning_factor_x = 1;
        sensor_mode_data.binning_factor_y = 1;
        break;
    }

    XCAM_ASSERT (aiq_context);
    const char *path_cpf = getenv ("AIQ_CPF_PATH");
    XCAM_FAIL_RETURN (
        WARNING,
        aiq_context->setup_analyzer (sensor_mode_data, path_cpf == NULL ? DEFAULT_AIQ_CPF_FILE : path_cpf),
        XCAM_RETURN_ERROR_UNKNOWN,
        "setup aiq 3a analyzer failed");

    SmartPtr<X3aAnalyzeTuner> analyzer = aiq_context->get_analyzer ();

    ret = analyzer->prepare_handlers ();
    XCAM_FAIL_RETURN (
        WARNING,
        ret == XCAM_RETURN_NO_ERROR,
        ret,
        "analyzer(aiq3alib) prepare handlers failed");

    ret = analyzer->init (width, height, framerate);
    XCAM_FAIL_RETURN (
        WARNING,
        ret == XCAM_RETURN_NO_ERROR,
        ret,
        "configure aiq 3a failed");

    ret = analyzer->start ();
    XCAM_FAIL_RETURN (
        WARNING,
        ret == XCAM_RETURN_NO_ERROR,
        ret,
        "start aiq 3a failed");

    aiq_context->set_size (width, height);

    return XCAM_RETURN_NO_ERROR;
}

static XCamReturn
xcam_set_3a_stats (XCam3AContext *context, XCam3AStats *stats, int64_t timestamp)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    XCam3AAiqContext *aiq_context = AIQ_CONTEXT_CAST (context);
    XCAM_ASSERT (aiq_context);

    SmartPtr<X3aAnalyzeTuner> analyzer = aiq_context->get_analyzer ();
    XCAM_ASSERT (analyzer.ptr ());
    XCAM_ASSERT (stats);

    if (!aiq_context->is_stats_pool_ready ()) {
        // init statistics queue
        XCAM_FAIL_RETURN (
            WARNING,
            aiq_context->setup_stats_pool (stats->info.bit_depth),
            XCAM_RETURN_ERROR_UNKNOWN,
            "aiq configure 3a failed on stats pool setup");
    }

    // Convert stats to atomisp_3a_stats;
    SmartPtr<X3aIspStatistics> isp_stats = aiq_context->get_stats_buffer ();
    if (!isp_stats.ptr ()) {
        XCAM_LOG_WARNING ("get stats bufffer failed or stopped");
        return XCAM_RETURN_ERROR_MEM;
    }

    struct atomisp_3a_statistics *raw_stats = isp_stats->get_isp_stats ();
    XCAM_ASSERT (raw_stats);

    translate_3a_stats (stats, raw_stats);
    isp_stats->set_timestamp (timestamp);

    ret = analyzer->push_3a_stats (isp_stats);
    if (ret != XCAM_RETURN_NO_ERROR) {
        XCAM_LOG_WARNING ("set 3a stats failed");
    }

    return ret;
}

static XCamReturn
xcam_update_common_params (XCam3AContext *context, XCamCommonParam *params)
{
    if (params) {
        SmartPtr<X3aAnalyzeTuner> analyzer = get_analyzer (context);
        XCAM_ASSERT (analyzer.ptr ());

        analyzer->update_common_parameters (*params);
    }
#if 0
    XCam3AAiqContext *aiq_context = AIQ_CONTEXT_CAST (context);
    aiq_context->update_brightness_result(params);
#endif
    return XCAM_RETURN_NO_ERROR;
}

static XCamReturn
xcam_analyze_awb (XCam3AContext *context, XCamAwbParam *params)
{
    if (params) {
        SmartPtr<X3aAnalyzeTuner> analyzer = get_analyzer (context);
        XCAM_ASSERT (analyzer.ptr ());

        analyzer->update_awb_parameters (*params);
    }
    return XCAM_RETURN_NO_ERROR;
}

static XCamReturn
xcam_analyze_ae (XCam3AContext *context, XCamAeParam *params)
{
    if (params) {
        SmartPtr<X3aAnalyzeTuner> analyzer = get_analyzer (context);
        XCAM_ASSERT (analyzer.ptr ());

        analyzer->update_ae_parameters (*params);
    }
    return XCAM_RETURN_NO_ERROR;
}

static XCamReturn
xcam_analyze_af (XCam3AContext *context, XCamAfParam *params)
{
    if (params) {
        SmartPtr<X3aAnalyzeTuner> analyzer = get_analyzer (context);
        XCAM_ASSERT (analyzer.ptr ());

        analyzer->update_af_parameters (*params);
    }
    return XCAM_RETURN_NO_ERROR;
}

static XCamReturn
xcam_combine_analyze_results (XCam3AContext *context, XCam3aResultHead *results[], uint32_t *res_count)
{
    XCam3AAiqContext *aiq_context = AIQ_CONTEXT_CAST (context);
    XCAM_ASSERT (aiq_context);
    X3aResultList aiq_results;
    uint32_t result_count = aiq_context->get_results (aiq_results);

    if (!result_count) {
        *res_count = 0;
        XCAM_LOG_DEBUG ("aiq wrapper combine with no result out");
        return XCAM_RETURN_NO_ERROR;
    }

    // mark as static
    static XCam3aResultHead *res_array[XCAM_3A_MAX_RESULT_COUNT];
    xcam_mem_clear (res_array);
    XCAM_ASSERT (result_count < XCAM_3A_MAX_RESULT_COUNT);

    // result_count may changed
    result_count = translate_3a_results_to_xcam (aiq_results, res_array, XCAM_3A_MAX_RESULT_COUNT);

    for (uint32_t i = 0; i < result_count; ++i) {
        results[i] = res_array[i];
    }
    *res_count = result_count;
    XCAM_ASSERT (result_count > 0);

    return XCAM_RETURN_NO_ERROR;
}

static void
xcam_free_results (XCam3aResultHead *results[], uint32_t res_count)
{
    for (uint32_t i = 0; i < res_count; ++i) {
        if (results[i])
            free_3a_result (results[i]);
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

