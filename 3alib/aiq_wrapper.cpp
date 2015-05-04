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

#define DEFAULT_AIQ_CPF_FILE       "/etc/atomisp/imx185.cpf"


using namespace XCam;

#define AIQ_CONTEXT_CAST(context)  ((XCam3AAiqContext*)(context))

class XCam3AAiqContext
    : public AnalyzerCallback
{
public:
    XCam3AAiqContext ();
    bool setup_analyzer (SmartPtr<IspController> &isp, const char *cpf);
    bool setup_stats_pool (uint32_t width, uint32_t height);
    SmartPtr<X3aAnalyzerAiq> &get_analyzer () {
        return _analyzer;
    }

    SmartPtr<X3aIspStatistics> get_stats_buffer ();
    uint32_t get_results (X3aResultList &results);

    // derive from AnalyzerCallback
    virtual void x3a_calculation_done (X3aAnalyzer *analyzer, X3aResultList &results);

private:
    XCAM_DEAD_COPY (XCam3AAiqContext);

private:
// members
    SmartPtr<X3aAnalyzerAiq>       _analyzer;
    SmartPtr<X3aStatisticsQueue>   _stats_pool;

    Mutex                          _result_mutex;
    X3aResultList                  _results;
};

XCam3AAiqContext::XCam3AAiqContext ()
{
}

bool
XCam3AAiqContext::setup_analyzer (SmartPtr<IspController> &isp, const char *cpf)
{
    XCAM_ASSERT (!_analyzer.ptr ());
    _analyzer = new X3aAnalyzerAiq (isp, cpf);
    XCAM_ASSERT (_analyzer.ptr ());
    _analyzer->set_results_callback (this);
    return true;
}

bool
XCam3AAiqContext::setup_stats_pool (uint32_t width, uint32_t height)
{
    VideoBufferInfo info;
    info.init (XCAM_PIX_FMT_SGRBG16, width, height);

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
XCam3AAiqContext::x3a_calculation_done (X3aAnalyzer *analyzer, X3aResultList &results)
{
    XCAM_UNUSED (analyzer);
    SmartLock  locker (_result_mutex);
    _results.insert (_results.end (), results.begin (), results.end ());
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

static SmartPtr<X3aAnalyzerAiq>
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
    if (aiq_context)
        delete aiq_context;
    return XCAM_RETURN_NO_ERROR;
}

static XCamReturn
xcam_configure_3a (XCam3AContext *context, uint32_t width, uint32_t height, double framerate)
{
    XCam3AAiqContext *aiq_context = AIQ_CONTEXT_CAST (context);
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    SmartPtr<IspController> isp;

    XCAM_ASSERT (aiq_context);
    XCAM_FAIL_RETURN (
        WARNING,
        aiq_context->setup_analyzer (isp, DEFAULT_AIQ_CPF_FILE),
        XCAM_RETURN_ERROR_UNKNOWN,
        "setup aiq 3a analyzer failed");

    SmartPtr<X3aAnalyzerAiq> analyzer = aiq_context->get_analyzer ();
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

    // init statistics queue
    XCAM_FAIL_RETURN (
        WARNING,
        aiq_context->setup_stats_pool (width, height),
        ret,
        "aiq configure 3a failed on stats pool setup");

    return XCAM_RETURN_NO_ERROR;
}

static XCamReturn
xcam_set_3a_stats (XCam3AContext *context, XCam3AStats *stats)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    XCam3AAiqContext *aiq_context = AIQ_CONTEXT_CAST (context);
    XCAM_ASSERT (aiq_context);

    SmartPtr<X3aAnalyzerAiq> analyzer = aiq_context->get_analyzer ();
    XCAM_ASSERT (analyzer.ptr ());
    XCAM_ASSERT (stats);

    // Convert stats to atomisp_3a_stats;
    SmartPtr<X3aIspStatistics> isp_stats = aiq_context->get_stats_buffer ();
    struct atomisp_3a_statistics *raw_stats = isp_stats->get_isp_stats ();
    XCAM_ASSERT (raw_stats);

    XCamAiq3A::translate_3a_stats (stats, raw_stats);

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
        SmartPtr<X3aAnalyzerAiq> analyzer = get_analyzer (context);
        XCAM_ASSERT (analyzer.ptr ());

        analyzer->update_common_parameters (*params);
    }
    return XCAM_RETURN_NO_ERROR;
}

static XCamReturn
xcam_analyze_awb (XCam3AContext *context, XCamAwbParam *params)
{
    if (params) {
        SmartPtr<X3aAnalyzerAiq> analyzer = get_analyzer (context);
        XCAM_ASSERT (analyzer.ptr ());

        analyzer->update_awb_parameters (*params);
    }
    return XCAM_RETURN_NO_ERROR;
}

static XCamReturn
xcam_analyze_ae (XCam3AContext *context, XCamAeParam *params)
{
    if (params) {
        SmartPtr<X3aAnalyzerAiq> analyzer = get_analyzer (context);
        XCAM_ASSERT (analyzer.ptr ());

        analyzer->update_ae_parameters (*params);
    }
    return XCAM_RETURN_NO_ERROR;
}

static XCamReturn
xcam_analyze_af (XCam3AContext *context, XCamAfParam *params)
{
    if (params) {
        SmartPtr<X3aAnalyzerAiq> analyzer = get_analyzer (context);
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
        XCAM_LOG_DEBUG ("aiq wrapper comible with no result out");
        return XCAM_RETURN_NO_ERROR;
    }

    // mark as static
    static XCam3aResultHead *res_array[XCAM_3A_LIB_MAX_RESULT_COUNT];
    xcam_mem_clear (res_array);
    XCAM_ASSERT (result_count < XCAM_3A_LIB_MAX_RESULT_COUNT);

    // result_count may changed
    result_count = XCamAiq3A::translate_3a_results_to_xcam (aiq_results, res_array, XCAM_3A_LIB_MAX_RESULT_COUNT);

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
            XCamAiq3A::free_3a_result (results[i]);
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

