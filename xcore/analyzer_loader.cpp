/*
 * analyzer_loader.cpp - analyzer loader
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

#include "analyzer_loader.h"
#include "handler_interface.h"
#include "x3a_result_factory.h"
#include "x3a_statistics_queue.h"
#include <dlfcn.h>

namespace XCam {

class DynamicAeHandler;
class DynamicAwbHandler;
class DynamicAfHandler;
class DynamicCommonHandler;

class DynamicAnalyzer
    : public X3aAnalyzer
{
public:
    DynamicAnalyzer (XCam3ADescription *desc);
    ~DynamicAnalyzer ();

public:
    XCamReturn analyze_ae (XCamAeParam &param);
    XCamReturn analyze_awb (XCamAwbParam &param);
    XCamReturn analyze_af (XCamAfParam &param);

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

    XCamReturn create_context ();
    void destroy_context ();

private:
    XCamReturn convert_results (XCam3aResultHead *from[], uint32_t from_count, X3aResultList &to);
    XCAM_DEAD_COPY (DynamicAnalyzer);

private:
    XCam3ADescription           *_desc;
    XCam3AContext               *_context;
    SmartPtr<X3aStats>           _cur_stats;
    SmartPtr<DynamicCommonHandler> _common_handler;
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
        AnalyzerHandler::HanlderLock lock(this);
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
        AnalyzerHandler::HanlderLock lock(this);
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
        AnalyzerHandler::HanlderLock lock(this);
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
        AnalyzerHandler::HanlderLock lock(this);
        return XCAM_RETURN_NO_ERROR;
    }

private:
    DynamicAnalyzer *_analyzer;
};

DynamicAnalyzer::DynamicAnalyzer (XCam3ADescription *desc)
    : X3aAnalyzer ("DynamicAnalyzer")
    , _desc (desc)
    , _context (NULL)
{
}

DynamicAnalyzer::~DynamicAnalyzer ()
{
    destroy_context ();
}

XCamReturn
DynamicAnalyzer::create_context ()
{
    XCam3AContext *context = NULL;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    XCAM_ASSERT (!_context);
    if ((ret = _desc->create_context (&context)) != XCAM_RETURN_NO_ERROR) {
        XCAM_LOG_WARNING ("dynamic 3a lib create context failed");
        return ret;
    }
    _context = context;
    return XCAM_RETURN_NO_ERROR;
}

void
DynamicAnalyzer::destroy_context ()
{
    if (_context && _desc && _desc->destroy_context) {
        _desc->destroy_context (_context);
        _context = NULL;
    }
}

XCamReturn
DynamicAnalyzer::analyze_ae (XCamAeParam &param)
{
    XCAM_ASSERT (_context);
    return _desc->analyze_ae (_context, &param);
}

XCamReturn
DynamicAnalyzer::analyze_awb (XCamAwbParam &param)
{
    XCAM_ASSERT (_context);
    return _desc->analyze_awb (_context, &param);
}

XCamReturn
DynamicAnalyzer::analyze_af (XCamAfParam &param)
{
    XCAM_ASSERT (_context);
    return _desc->analyze_af (_context, &param);
}

SmartPtr<AeHandler>
DynamicAnalyzer::create_ae_handler ()
{
    return new DynamicAeHandler (this);
}

SmartPtr<AwbHandler>
DynamicAnalyzer::create_awb_handler ()
{
    return new DynamicAwbHandler (this);
}

SmartPtr<AfHandler>
DynamicAnalyzer::create_af_handler ()
{
    return new DynamicAfHandler (this);
}

SmartPtr<CommonHandler>
DynamicAnalyzer::create_common_handler ()
{
    if (_common_handler.ptr())
        return _common_handler;

    _common_handler = new DynamicCommonHandler (this);
    return _common_handler;
}

XCamReturn
DynamicAnalyzer::internal_init (uint32_t width, uint32_t height, double framerate)
{
    XCAM_UNUSED (width);
    XCAM_UNUSED (height);
    XCAM_UNUSED (framerate);
    return create_context ();
}

XCamReturn
DynamicAnalyzer::internal_deinit ()
{
    destroy_context ();
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
DynamicAnalyzer::configure_3a ()
{
    uint32_t width = get_width ();
    uint32_t height = get_height ();
    double framerate = get_framerate ();
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    XCAM_ASSERT (_context);

    XCAM_FAIL_RETURN (WARNING,
                      ret = _desc->configure_3a (_context, width, height, framerate),
                      ret,
                      "dynamic analyzer configure 3a failed");

    return XCAM_RETURN_NO_ERROR;
}
XCamReturn
DynamicAnalyzer::pre_3a_analyze (SmartPtr<X3aStats> &stats)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    XCamCommonParam common_params = _common_handler->get_params_unlock ();

    XCAM_ASSERT (_context);
    _cur_stats = stats;
    ret = _desc->set_3a_stats (_context, stats->get_stats());
    XCAM_FAIL_RETURN (WARNING,
                      ret == XCAM_RETURN_NO_ERROR,
                      ret,
                      "dynamic analyzer set_3a_stats failed");

    ret = _desc->update_common_params (_context, &common_params);
    XCAM_FAIL_RETURN (WARNING,
                      ret == XCAM_RETURN_NO_ERROR,
                      ret,
                      "dynamic analyzer update common params failed");

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
DynamicAnalyzer::post_3a_analyze (X3aResultList &results)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    XCam3aResultHead *res_array[XCAM_3A_LIB_MAX_RESULT_COUNT];
    uint32_t res_count = 0;

    xcam_mem_clear (res_array);
    XCAM_ASSERT (_context);
    ret = _desc->combine_analyze_results (_context, res_array, &res_count);
    XCAM_FAIL_RETURN (WARNING,
                      ret == XCAM_RETURN_NO_ERROR,
                      ret,
                      "dynamic analyzer combine_analyze_results failed");

    _cur_stats.release ();

    if (res_count) {
        ret = convert_results (res_array, res_count, results);
        XCAM_FAIL_RETURN (WARNING,
                          ret == XCAM_RETURN_NO_ERROR,
                          ret,
                          "dynamic analyzer convert_results failed");
        _desc->free_results (res_array, res_count);
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
DynamicAnalyzer::convert_results (XCam3aResultHead *from[], uint32_t from_count, X3aResultList &to)
{
    for (uint32_t i = 0; i < from_count; ++i) {
        SmartPtr<X3aResult> standard_res =
            X3aResultFactory::instance ()->create_3a_result (from[i]);
        to.push_back (standard_res);
    }

    return XCAM_RETURN_NO_ERROR;
}

AnalyzerLoader::AnalyzerLoader (const char *lib_path)
    : _path (NULL)
    , _handle (NULL)
{
    XCAM_ASSERT (lib_path);
    _path = strdup (lib_path);
}

AnalyzerLoader::~AnalyzerLoader ()
{
    close_handle ();
    if (_path)
        xcam_free (_path);
}

SmartPtr<X3aAnalyzer>
AnalyzerLoader::load_analyzer ()
{
    SmartPtr<X3aAnalyzer> analyzer;
    XCam3ADescription *desc = NULL;
    const char *symbol = XCAM_3A_LIB_DESCRIPTION;
    if (!open_handle ()) {
        XCAM_LOG_WARNING ("open dynamic lib:%s failed", XCAM_STR (_path));
        return NULL;
    }

    desc = get_symbol (symbol);
    if (!desc) {
        XCAM_LOG_WARNING ("get symbol(%s) from lib:%s failed", symbol, XCAM_STR (_path));
        close_handle ();
        return NULL;
    }

    XCAM_LOG_DEBUG ("got symbols(%s) from lib(%s)", symbol, XCAM_STR (_path));

    analyzer = new DynamicAnalyzer (desc);
    if (!analyzer.ptr ()) {
        XCAM_LOG_WARNING ("get symbol(%s) from lib:%s failed", symbol, XCAM_STR (_path));
        close_handle ();
        return NULL;
    }

    XCAM_LOG_INFO ("analyzer(%s) created from 3a lib(%s)", XCAM_STR (analyzer->get_name()), XCAM_STR (_path));
    return analyzer;
}

bool
AnalyzerLoader::open_handle ()
{
    void *handle = NULL;

    if (_handle != NULL)
        return true;

    handle = dlopen (_path, RTLD_LAZY);
    if (!handle) {
        XCAM_LOG_DEBUG (
            "open user-defined 3a lib(%s) failed, reason:%s",
            XCAM_STR (_path), dlerror ());
        return false;
    }
    _handle = handle;
    return true;
}

XCam3ADescription *
AnalyzerLoader::get_symbol (const char *symbol)
{
    XCam3ADescription *desc = NULL;

    XCAM_ASSERT (_handle);
    XCAM_ASSERT (symbol);
    desc = (XCam3ADescription *)dlsym (_handle, symbol);
    if (!desc) {
        XCAM_LOG_DEBUG ("get symbol(%s) failed from lib(%s), reason:%s", symbol, XCAM_STR (_path), dlerror ());
        return NULL;
    }
    if (desc->version < XCAM_VERSION) {
        XCAM_LOG_DEBUG ("get symbol(%s) failed. version is:0x%04x, but expect:0x%04x",
                        symbol, desc->version, XCAM_VERSION);
        return NULL;
    }
    if (desc->size < sizeof (XCam3ADescription)) {
        XCAM_LOG_DEBUG ("get symbol(%s) failed, XCam3ADescription size is:%d, but expect:%d",
                        symbol, desc->size, sizeof (XCam3ADescription));
        return NULL;
    }

    if (!desc->create_context || !desc->destroy_context ||
            !desc->configure_3a || !desc->set_3a_stats ||
            !desc->analyze_awb || !desc->analyze_ae ||
            !desc->analyze_af || !desc->combine_analyze_results ||
            !desc->free_results) {
        XCAM_LOG_DEBUG ("some functions in symbol(%s) not set from lib(%s)", symbol, XCAM_STR (_path));
        return NULL;
    }
    return desc;
}

bool
AnalyzerLoader::close_handle ()
{
    if (!_handle)
        return true;
    dlclose (_handle);
    _handle = NULL;
    return true;
}

};
