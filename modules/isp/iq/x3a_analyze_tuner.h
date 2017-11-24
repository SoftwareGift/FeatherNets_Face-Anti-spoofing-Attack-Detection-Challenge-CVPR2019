/*
 * x3a_analyze_tuner.h - x3a Common IQ tuner
 *
 *  Copyright (c) 2014-2015 Intel Corporation
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
 * Author: Zong Wei <wei.zong@intel.com>
 */

#ifndef XCAM_3A_ANALYZE_TUNER_H
#define XCAM_3A_ANALYZE_TUNER_H

namespace XCam {

class X3aCiqTuningHandler;
class X3aAnalyzer;

class X3aAnalyzeTuner
    : public X3aAnalyzer
    , public AnalyzerCallback
{
    typedef std::list<SmartPtr<X3aCiqTuningHandler>> X3aCiqTuningHandlerList;

public:
    explicit X3aAnalyzeTuner ();
    virtual ~X3aAnalyzeTuner ();

    void enable_handler ();
    void set_analyzer (SmartPtr<X3aAnalyzer> &analyzer);

    XCamReturn analyze_ae (XCamAeParam &param);
    XCamReturn analyze_awb (XCamAwbParam &param);
    XCamReturn analyze_af (XCamAfParam &param);
    XCamReturn analyze_common (XCamCommonParam &param);

protected:
    virtual SmartPtr<AeHandler> create_ae_handler ();
    virtual SmartPtr<AwbHandler> create_awb_handler ();
    virtual SmartPtr<AfHandler> create_af_handler ();
    virtual SmartPtr<CommonHandler> create_common_handler ();

    virtual XCamReturn internal_init (uint32_t width, uint32_t height, double framerate);
    virtual XCamReturn internal_deinit ();

    virtual XCamReturn configure_3a ();
    virtual XCamReturn pre_3a_analyze (SmartPtr<X3aStats> &stats);
    virtual XCamReturn post_3a_analyze (X3aResultList &results);

    // derive from AnalyzerCallback
    virtual void x3a_calculation_done (XAnalyzer *analyzer, X3aResultList &results);

private:
    XCAM_DEAD_COPY (X3aAnalyzeTuner);

    XCamReturn create_tuning_handlers ();
    bool add_handler (SmartPtr<X3aCiqTuningHandler> &handler);

protected:

private:
    SmartPtr<X3aAnalyzer> _analyzer;
    X3aCiqTuningHandlerList _handlers;
    SmartPtr<X3aStats> _stats;
    X3aResultList _results;
};

class X3aCiqTuningAeHandler
    : public AeHandler
{
public:
    explicit X3aCiqTuningAeHandler (X3aAnalyzeTuner *analyzer)
        : _analyzer (analyzer)
    {}
    virtual XCamReturn analyze (X3aResultList &output) {
        XCAM_UNUSED (output);
        AnalyzerHandler::HandlerLock lock(this);
        XCamAeParam param = this->get_params_unlock ();
        return _analyzer->analyze_ae (param);
    }

private:
    X3aAnalyzeTuner *_analyzer;
};

class X3aCiqTuningAwbHandler
    : public AwbHandler
{
public:
    explicit X3aCiqTuningAwbHandler (X3aAnalyzeTuner *analyzer)
        : _analyzer (analyzer)
    {}
    virtual XCamReturn analyze (X3aResultList &output) {
        XCAM_UNUSED (output);
        AnalyzerHandler::HandlerLock lock(this);
        XCamAwbParam param = this->get_params_unlock ();
        return _analyzer->analyze_awb (param);
    }

private:
    X3aAnalyzeTuner *_analyzer;
};

class X3aCiqTuningAfHandler
    : public AfHandler
{
public:
    explicit X3aCiqTuningAfHandler (X3aAnalyzeTuner *analyzer)
        : _analyzer (analyzer)
    {}
    virtual XCamReturn analyze (X3aResultList &output) {
        XCAM_UNUSED (output);
        AnalyzerHandler::HandlerLock lock(this);
        XCamAfParam param = this->get_params_unlock ();
        return _analyzer->analyze_af (param);
    }

private:
    X3aAnalyzeTuner *_analyzer;
};

class X3aCiqTuningCommonHandler
    : public CommonHandler
{
public:
    explicit X3aCiqTuningCommonHandler (X3aAnalyzeTuner *analyzer)
        : _analyzer (analyzer)
    {}
    virtual XCamReturn analyze (X3aResultList &output) {
        XCAM_UNUSED (output);
        AnalyzerHandler::HandlerLock lock(this);
        XCamCommonParam param = this->get_params_unlock ();
        return _analyzer->analyze_common (param);
    }

private:
    X3aAnalyzeTuner *_analyzer;
};

};
#endif // XCAM_3A_ANALYZE_TUNER_H
