/*
 * dynamic_analyzer.h - dynamic analyzer
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
 *         Jia Meng <jia.meng@intel.com>
 */
#ifndef XCAM_DYNAMIC_ANALYZER_H
#define XCAM_DYNAMIC_ANALYZER_H

#include <base/xcam_3a_description.h>
#include "x3a_analyzer.h"
#include "x3a_stats_pool.h"
#include "handler_interface.h"
#include "x3a_result_factory.h"
#include "analyzer_loader.h"

namespace XCam {

class DynamicAeHandler;
class DynamicAwbHandler;
class DynamicAfHandler;
class DynamicCommonHandler;

class DynamicAnalyzer
    : public X3aAnalyzer
{
public:
    DynamicAnalyzer (XCam3ADescription *desc, SmartPtr<AnalyzerLoader> &loader, const char *name = "DynamicAnalyzer");
    ~DynamicAnalyzer ();

    virtual XCamReturn configure_3a ();
    virtual XCamReturn analyze_ae (XCamAeParam &param);
    virtual XCamReturn analyze_awb (XCamAwbParam &param);
    virtual XCamReturn analyze_af (XCamAfParam &param);

protected:
    virtual SmartPtr<AeHandler> create_ae_handler ();
    virtual SmartPtr<AwbHandler> create_awb_handler ();
    virtual SmartPtr<AfHandler> create_af_handler ();
    virtual SmartPtr<CommonHandler> create_common_handler ();
    virtual XCamReturn internal_init (uint32_t width, uint32_t height, double framerate);
    virtual XCamReturn internal_deinit ();

    virtual XCamReturn pre_3a_analyze (SmartPtr<X3aStats> &stats);
    virtual XCamReturn post_3a_analyze (X3aResultList &results);

    XCamReturn create_context ();
    void destroy_context ();

    const XCamCommonParam get_common_params ();
    SmartPtr<X3aStats> get_cur_stats () const {
        return _cur_stats;
    }
    XCamReturn convert_results (XCam3aResultHead *from[], uint32_t from_count, X3aResultList &to);

private:
    XCAM_DEAD_COPY (DynamicAnalyzer);

private:
    XCam3ADescription                  *_desc;
    XCam3AContext                      *_context;
    SmartPtr<X3aStats>                 _cur_stats;
    SmartPtr<DynamicCommonHandler>     _common_handler;
    SmartPtr<AnalyzerLoader>           _loader;
};

class DynamicAeHandler
    : public AeHandler
{
public:
    explicit DynamicAeHandler (DynamicAnalyzer *analyzer)
        : _analyzer (analyzer)
    {}
    virtual XCamReturn analyze (X3aResultList &output) {
        XCAM_UNUSED (output);
        AnalyzerHandler::HandlerLock lock(this);
        XCamAeParam param = this->get_params_unlock ();
        return _analyzer->analyze_ae (param);
    }

private:
    DynamicAnalyzer *_analyzer;
};

class DynamicAwbHandler
    : public AwbHandler
{
public:
    explicit DynamicAwbHandler (DynamicAnalyzer *analyzer)
        : _analyzer (analyzer)
    {}
    virtual XCamReturn analyze (X3aResultList &output) {
        XCAM_UNUSED (output);
        AnalyzerHandler::HandlerLock lock(this);
        XCamAwbParam param = this->get_params_unlock ();
        return _analyzer->analyze_awb (param);
    }

private:
    DynamicAnalyzer *_analyzer;
};

class DynamicAfHandler
    : public AfHandler
{
public:
    explicit DynamicAfHandler (DynamicAnalyzer *analyzer)
        : _analyzer (analyzer)
    {}
    virtual XCamReturn analyze (X3aResultList &output) {
        XCAM_UNUSED (output);
        AnalyzerHandler::HandlerLock lock(this);
        XCamAfParam param = this->get_params_unlock ();
        return _analyzer->analyze_af (param);
    }

private:
    DynamicAnalyzer *_analyzer;
};

class DynamicCommonHandler
    : public CommonHandler
{
    friend class DynamicAnalyzer;
public:
    explicit DynamicCommonHandler (DynamicAnalyzer *analyzer)
        : _analyzer (analyzer)
    {}
    virtual XCamReturn analyze (X3aResultList &output) {
        XCAM_UNUSED (output);
        AnalyzerHandler::HandlerLock lock(this);
        return XCAM_RETURN_NO_ERROR;
    }

private:
    DynamicAnalyzer *_analyzer;
};
}

#endif //XCAM_DYNAMIC_ANALYZER_H
