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

#ifndef XCAM_HYBRID_ANALYZER_H
#define XCAM_HYBRID_ANALYZER_H

#include "dynamic_analyzer.h"

namespace XCam {
class IspController;
class X3aAnalyzerAiq;
class X3aStatisticsQueue;
class X3aIspStatistics;

class HybridAnalyzer
    : public DynamicAnalyzer
    , public AnalyzerCallback
{
public:
    explicit HybridAnalyzer (XCam3ADescription *desc,
                             SmartPtr<AnalyzerLoader> &loader,
                             SmartPtr<IspController> &isp,
                             const char *cpf_path);
    ~HybridAnalyzer ();

    virtual XCamReturn analyze_ae (XCamAeParam &param);
    virtual XCamReturn analyze_awb (XCamAwbParam &param);
    virtual XCamReturn analyze_af (XCamAfParam &param);

    virtual void x3a_calculation_done (X3aAnalyzer *analyzer, X3aResultList &results);
    virtual void x3a_calculation_failed (X3aAnalyzer *analyzer, int64_t timestamp, const char *msg);

protected:
    virtual XCamReturn internal_init (uint32_t width, uint32_t height, double framerate);
    virtual XCamReturn internal_deinit ();

    virtual XCamReturn configure_3a ();
    virtual XCamReturn pre_3a_analyze (SmartPtr<X3aStats> &stats);
    virtual XCamReturn post_3a_analyze (X3aResultList &results);

private:
    XCAM_DEAD_COPY (HybridAnalyzer);
    XCamReturn setup_stats_pool (const XCam3AStats *stats);
    SmartPtr<X3aIspStatistics> convert_to_isp_stats (SmartPtr<X3aStats>& stats);

    SmartPtr<IspController>       _isp;
    const char                    *_cpf_path;
    SmartPtr<X3aAnalyzerAiq>      _analyzer_aiq;
    SmartPtr<X3aStatisticsQueue>  _stats_pool;
};

}

#endif //XCAM_HYBRID_ANALYZER_H
