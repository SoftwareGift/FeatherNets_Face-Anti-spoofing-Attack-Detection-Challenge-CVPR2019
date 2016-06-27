/*
 * hybrid_analyzer.h - hybrid analyzer
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

#include "hybrid_analyzer.h"
#include "isp_controller.h"
#include "x3a_analyzer_aiq.h"
#include "x3a_statistics_queue.h"
#include "aiq3a_utils.h"
#include <math.h>

namespace XCam {
HybridAnalyzer::HybridAnalyzer (XCam3ADescription *desc,
                                   SmartPtr<AnalyzerLoader> &loader,
                                   SmartPtr<IspController> &isp,
                                   const char *cpf_path)
    : DynamicAnalyzer (desc, loader, "HybridAnalyzer")
    , _isp (isp)
    , _cpf_path (cpf_path)
{
    _analyzer_aiq = new X3aAnalyzerAiq (isp, cpf_path);
    XCAM_ASSERT (_analyzer_aiq.ptr ());
    _analyzer_aiq->prepare_handlers ();
    _analyzer_aiq->set_results_callback (this);
    _analyzer_aiq->set_sync_mode (true);
}

XCamReturn
HybridAnalyzer::internal_init (uint32_t width, uint32_t height, double framerate)
{
    if (_analyzer_aiq->init (width, height, framerate) != XCAM_RETURN_NO_ERROR) {
        return XCAM_RETURN_ERROR_AIQ;
    }

    return create_context ();
}

XCamReturn
HybridAnalyzer::internal_deinit ()
{
    if (_analyzer_aiq->deinit () != XCAM_RETURN_NO_ERROR) {
        return XCAM_RETURN_ERROR_AIQ;
    }

    return DynamicAnalyzer::internal_deinit ();
}

XCamReturn
HybridAnalyzer::setup_stats_pool (const XCam3AStats *stats)
{
    XCAM_ASSERT (stats);

    XCam3AStatsInfo stats_info = stats->info;
    struct atomisp_grid_info grid_info;
    grid_info.enable = 1;
    grid_info.use_dmem = 0;
    grid_info.has_histogram = 0;
    grid_info.width = stats_info.width;
    grid_info.height = stats_info.height;
    grid_info.aligned_width = stats_info.aligned_width;
    grid_info.aligned_height = stats_info.aligned_height;
    grid_info.bqs_per_grid_cell = stats_info.grid_pixel_size >> 1;
    grid_info.deci_factor_log2 = log2 (grid_info.bqs_per_grid_cell);
    grid_info.elem_bit_depth = stats_info.bit_depth;

    _stats_pool = new X3aStatisticsQueue;
    XCAM_ASSERT (_stats_pool.ptr ());

    _stats_pool->set_grid_info (grid_info);
    if (!_stats_pool->reserve (6)) {
        XCAM_LOG_WARNING ("setup_stats_pool failed to reserve stats buffer.");
        return XCAM_RETURN_ERROR_MEM;
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
HybridAnalyzer::configure_3a ()
{
    if (_analyzer_aiq->start () != XCAM_RETURN_NO_ERROR) {
        return XCAM_RETURN_ERROR_AIQ;
    }

    return DynamicAnalyzer::configure_3a ();
}

XCamReturn
HybridAnalyzer::pre_3a_analyze (SmartPtr<X3aStats> &stats)
{
    _analyzer_aiq->update_common_parameters (get_common_params ());

    return DynamicAnalyzer::pre_3a_analyze (stats);
}

SmartPtr<X3aIspStatistics>
HybridAnalyzer::convert_to_isp_stats (SmartPtr<X3aStats>& stats)
{
    SmartPtr<X3aIspStatistics> isp_stats =
        _stats_pool->get_buffer (_stats_pool).dynamic_cast_ptr<X3aIspStatistics> ();

    XCAM_FAIL_RETURN (
        WARNING,
        isp_stats.ptr (),
        NULL,
        "get isp stats buffer failed");

    struct atomisp_3a_statistics *to = isp_stats->get_isp_stats ();
    XCam3AStats *from = stats->get_stats ();
    translate_3a_stats (from, to);
    isp_stats->set_timestamp (stats->get_timestamp ());
    return isp_stats;
}

XCamReturn
HybridAnalyzer::post_3a_analyze (X3aResultList &results)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    SmartPtr<X3aStats> stats = get_cur_stats ();

    if ((ret = DynamicAnalyzer::post_3a_analyze (results)) != XCAM_RETURN_NO_ERROR) {
        return ret;
    }

    for (X3aResultList::iterator i_res = results.begin ();
            i_res != results.end (); ++i_res) {
        SmartPtr<X3aResult> result = *i_res;

        switch (result->get_type ()) {
        case XCAM_3A_RESULT_EXPOSURE: {
            XCam3aResultExposure *res = (XCam3aResultExposure *) result->get_ptr ();
            _analyzer_aiq->set_ae_mode (XCAM_AE_MODE_MANUAL);
            _analyzer_aiq->set_ae_manual_exposure_time (res->exposure_time);
            _analyzer_aiq->set_ae_manual_analog_gain (res->analog_gain);
            break;
        }
        case XCAM_3A_RESULT_WHITE_BALANCE: {
            _analyzer_aiq->set_awb_mode (XCAM_AWB_MODE_MANUAL);
            XCam3aResultWhiteBalance *res = (XCam3aResultWhiteBalance *) result->get_ptr ();
            _analyzer_aiq->set_awb_manual_gain (res->gr_gain, res->r_gain, res->b_gain, res->gb_gain);
            break;
        }
        default:
            break;
        }
    }
    results.clear ();

    SmartPtr<X3aIspStatistics> isp_stats = stats.dynamic_cast_ptr<X3aIspStatistics> ();
    if (!isp_stats.ptr ()) {
        if (!_stats_pool.ptr () && setup_stats_pool (stats->get_stats ()) != XCAM_RETURN_NO_ERROR)
            return XCAM_RETURN_ERROR_MEM;
        isp_stats = convert_to_isp_stats (stats);
    }
    return _analyzer_aiq->push_3a_stats (isp_stats);
}

XCamReturn
HybridAnalyzer::analyze_ae (XCamAeParam &param)
{
    if (!_analyzer_aiq->update_ae_parameters (param))
        return XCAM_RETURN_ERROR_AIQ;

    return DynamicAnalyzer::analyze_ae (param);
}

XCamReturn
HybridAnalyzer::analyze_awb (XCamAwbParam &param)
{
    if (!_analyzer_aiq->update_awb_parameters (param))
        return XCAM_RETURN_ERROR_AIQ;

    return DynamicAnalyzer::analyze_awb (param);
}

XCamReturn
HybridAnalyzer::analyze_af (XCamAfParam &param)
{
    if (!_analyzer_aiq->update_af_parameters (param))
        return XCAM_RETURN_ERROR_AIQ;

    return DynamicAnalyzer::analyze_af (param);
}

void
HybridAnalyzer::x3a_calculation_done (XAnalyzer *analyzer, X3aResultList &results)
{
    XCAM_UNUSED (analyzer);

    static XCam3aResultHead *res_heads[XCAM_3A_MAX_RESULT_COUNT];
    xcam_mem_clear (res_heads);
    XCAM_ASSERT (results.size () < XCAM_3A_MAX_RESULT_COUNT);

    uint32_t result_count = translate_3a_results_to_xcam (results,
                            res_heads, XCAM_3A_MAX_RESULT_COUNT);
    convert_results (res_heads, result_count, results);
    for (uint32_t i = 0; i < result_count; ++i) {
        if (res_heads[i])
            free_3a_result (res_heads[i]);
    }

    notify_calculation_done (results);
}

HybridAnalyzer::~HybridAnalyzer ()
{
    destroy_context ();
}

void
HybridAnalyzer::x3a_calculation_failed (XAnalyzer *analyzer, int64_t timestamp, const char *msg)
{
    XCAM_UNUSED (analyzer);
    notify_calculation_failed (NULL, timestamp, msg);
}
}
