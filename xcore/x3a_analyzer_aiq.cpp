/*
 * x3a_analyzer_aiq.h - 3a analyzer from AIQ
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
 * Author: Wind Yuan <feng.yuan@intel.com>
 */

#include "x3a_analyzer_aiq.h"
#include "aiq_handler.h"
#include "isp_controller.h"
#include "xcam_cpf_reader.h"
#include "ia_types.h"

namespace XCam {

class CpfReader {
public:
    explicit CpfReader (const char *name);
    ~CpfReader();
    bool read (ia_binary_data &binary);
private:
    XCamCpfBlob *_aiq_cpf;
    char *_name;
};

CpfReader::CpfReader (const char *name)
    : _name (strdup(name))
{
    _aiq_cpf = xcam_cpf_blob_new ();
    XCAM_ASSERT (name);
}
CpfReader::~CpfReader()
{
    if (_aiq_cpf)
        xcam_cpf_blob_free (_aiq_cpf);
    if (_name)
        xcam_free (_name);
}

bool CpfReader::read (ia_binary_data &binary)
{
    if (!xcam_cpf_read (_name, _aiq_cpf, NULL)) {
        XCAM_LOG_ERROR ("parse CPF(%s) failed", XCAM_STR (_name));
        return false;
    }
    binary.data  = _aiq_cpf->data;
    binary.size = _aiq_cpf->size;
    XCAM_LOG_INFO ("read cpf(%s) ok", XCAM_STR (_name));
    return true;
}

X3aAnalyzerAiq::X3aAnalyzerAiq (SmartPtr<IspController> &isp, const char *cpf_path)
    : X3aAnalyzer ("X3aAnalyzerAiq")
    , _isp (isp)
    , _cpf_path (NULL)
{
    if (cpf_path)
        _cpf_path = strdup (cpf_path);

    _aiq_compositor = new AiqCompositor ();
    XCAM_ASSERT (_aiq_compositor.ptr());
    xcam_mem_clear (_sensor_mode_data);

    XCAM_LOG_DEBUG ("X3aAnalyzerAiq constructed");
}

X3aAnalyzerAiq::~X3aAnalyzerAiq()
{
    if (_cpf_path)
        xcam_free (_cpf_path);

    XCAM_LOG_DEBUG ("~X3aAnalyzerAiq destructed");
}

SmartPtr<AeHandler>
X3aAnalyzerAiq::create_ae_handler ()
{

    SmartPtr<AiqAeHandler> ae_handler = new AiqAeHandler (_aiq_compositor);
    _aiq_compositor->set_ae_handler (ae_handler);
    return ae_handler;
}

SmartPtr<AwbHandler>
X3aAnalyzerAiq::create_awb_handler ()
{
    SmartPtr<AiqAwbHandler> awb_handler = new AiqAwbHandler (_aiq_compositor);
    _aiq_compositor->set_awb_handler (awb_handler);
    return awb_handler;
}

SmartPtr<AfHandler>
X3aAnalyzerAiq::create_af_handler ()
{

    SmartPtr<AiqAfHandler> af_handler = new AiqAfHandler (_aiq_compositor);
    _aiq_compositor->set_af_handler (af_handler);
    return af_handler;
}

SmartPtr<CommonHandler>
X3aAnalyzerAiq::create_common_handler ()
{
    SmartPtr<AiqCommonHandler> common_handler = new AiqCommonHandler (_aiq_compositor);
    _aiq_compositor->set_common_handler (common_handler);
    return common_handler;
}

XCamReturn
X3aAnalyzerAiq::internal_init (uint32_t width, uint32_t height, double framerate)
{
    XCAM_UNUSED (framerate);
    XCAM_ASSERT (_cpf_path);
    CpfReader reader (_cpf_path);
    ia_binary_data binary;

    XCAM_ASSERT (_aiq_compositor.ptr());

    xcam_mem_clear (binary);
    XCAM_FAIL_RETURN (
        ERROR,
        reader.read(binary),
        XCAM_RETURN_ERROR_AIQ,
        "read cpf file(%s) failed", _cpf_path);

    _aiq_compositor->set_size (width, height);
    XCAM_FAIL_RETURN (
        ERROR,
        _aiq_compositor->open (binary),
        XCAM_RETURN_ERROR_AIQ,
        "AIQ open failed");

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
X3aAnalyzerAiq::internal_deinit ()
{
    if (_aiq_compositor.ptr ())
        _aiq_compositor->close ();

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
X3aAnalyzerAiq::configure_3a ()
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    X3aResultList first_results;
    struct atomisp_sensor_mode_data sensor_mode_data;

    XCAM_ASSERT (_isp.ptr());
    xcam_mem_clear (sensor_mode_data);

    ret = _isp->get_sensor_mode_data (sensor_mode_data);
    XCAM_FAIL_RETURN (WARNING, ret == XCAM_RETURN_NO_ERROR, ret, "get sensor mode data failed");

    if (!_aiq_compositor->set_sensor_mode_data (&sensor_mode_data)) {
        XCAM_LOG_WARNING ("AIQ configure 3a failed");
        return XCAM_RETURN_ERROR_AIQ;
    }

    XCAM_LOG_DEBUG ("X3aAnalyzerAiq got sensor mode data, coarse_time_min:%u, "
                    "coarse_time_max_margin:%u, "
                    "fine_time_min:%u, fine_time_max_margin:%u, "
                    "fine_time_def:%u, "
                    "frame_length_lines:%u, line_length_pck:%u, "
                    "vt_pix_clk_freq_mhz:%u, "
                    "crop_horizontal_start:%u, crop_vertical_start:%u, "
                    "crop_horizontal_end:%u, crop_vertical_end:%u, "
                    "output_width:%u, output_height:%u, "
                    "binning_factor_x:%u, binning_factor_y:%u",
                    sensor_mode_data.coarse_integration_time_min,
                    sensor_mode_data.coarse_integration_time_max_margin,
                    sensor_mode_data.fine_integration_time_min,
                    sensor_mode_data.fine_integration_time_max_margin,
                    sensor_mode_data.fine_integration_time_def,
                    sensor_mode_data.frame_length_lines,
                    sensor_mode_data.line_length_pck,
                    sensor_mode_data.vt_pix_clk_freq_mhz,
                    sensor_mode_data.crop_horizontal_start,
                    sensor_mode_data.crop_vertical_start,
                    sensor_mode_data.crop_horizontal_end,
                    sensor_mode_data.crop_vertical_end,
                    sensor_mode_data.output_width,
                    sensor_mode_data.output_height,
                    (uint32_t)sensor_mode_data.binning_factor_x,
                    (uint32_t)sensor_mode_data.binning_factor_y);

    // initialize ae and awb
    get_ae_handler ()->analyze (first_results);
    get_awb_handler ()->analyze (first_results);

    ret = _aiq_compositor->integrate (first_results);
    XCAM_FAIL_RETURN (WARNING, ret == XCAM_RETURN_NO_ERROR, ret, "AIQ configure_3a failed on integrate results");

    if (!first_results.empty()) {
        notify_calculation_done (first_results);
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
X3aAnalyzerAiq::pre_3a_analyze (SmartPtr<X3aStats> &stats)
{
    SmartPtr<X3aIspStatistics> isp_stats = stats.dynamic_cast_ptr<X3aIspStatistics> ();

    XCAM_ASSERT (isp_stats.ptr ());
    if (!_aiq_compositor->set_3a_stats (isp_stats)) {
        XCAM_LOG_WARNING ("Aiq compositor set 3a stats failed");
        return XCAM_RETURN_ERROR_UNKNOWN;
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
X3aAnalyzerAiq::post_3a_analyze (X3aResultList &results)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    ret = _aiq_compositor->integrate (results);
    XCAM_FAIL_RETURN (WARNING, ret == XCAM_RETURN_NO_ERROR, ret, "AIQ integrate 3A results failed");

    return XCAM_RETURN_NO_ERROR;
}

};
