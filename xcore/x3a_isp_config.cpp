/*
 * x3a_isp_config.h - 3A ISP config
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

#include "x3a_isp_config.h"
#include "isp_config_translator.h"

namespace XCam {

void AtomIspConfigContent::clear ()
{
    memset (this, 0, sizeof (AtomIspConfigContent));
}

void
AtomIspConfigContent::copy (const struct atomisp_parameters &config)
{
    xcam_mem_clear (isp_config);
    if (config.wb_config) {
        wb = *config.wb_config;
        isp_config.wb_config = &wb;
    }
    if (config.cc_config) {
        cc = *config.cc_config;
        isp_config.cc_config = &cc;
    }
    if (config.tnr_config) {
        tnr = *config.tnr_config;
        isp_config.tnr_config = &tnr;
    }
    if (config.ecd_config) {
        ecd_config = *config.ecd_config;
        isp_config.ecd_config = &ecd_config;
    }
    if (config.ynr_config) {
        ynr = *config.ynr_config;
        isp_config.ynr_config = &ynr;
    }
    if (config.fc_config) {
        fc_config = *config.fc_config;
        isp_config.fc_config = &fc_config;
    }
    if (config.cnr_config) {
        cnr = *config.cnr_config;
        isp_config.cnr_config = &cnr;
    }
    if (config.macc_config) {
        macc_config = *config.macc_config;
        isp_config.macc_config = &macc_config;
    }
    if (config.ctc_config) {
        ctc_config = *config.ctc_config;
        isp_config.ctc_config = &ctc_config;
    }
    if (config.formats_config) {
        formats = *config.formats_config;
        isp_config.formats_config = &formats;
    }
    if (config.aa_config) {
        aa = *config.aa_config;
        isp_config.aa_config = &aa;
    }
    if (config.baa_config) {
        baa = *config.baa_config;
        isp_config.baa_config = &baa;
    }
    if (config.ce_config) {
        ce = *config.ce_config;
        isp_config.ce_config = &ce;
    }
    if (config.dvs_6axis_config) {
        dvs_6axis = *config.dvs_6axis_config;
        isp_config.dvs_6axis_config = &dvs_6axis;
    }
    if (config.ob_config) {
        ob = *config.ob_config;
        isp_config.ob_config = &ob;
    }
    if (config.nr_config) {
        nr = *config.nr_config;
        isp_config.nr_config = &nr;
    }
    if (config.dp_config) {
        dp = *config.dp_config;
        isp_config.dp_config = &dp;
    }
    if (config.ee_config) {
        ee = *config.ee_config;
        isp_config.ee_config = &ee;
    }
    if (config.de_config) {
        de = *config.de_config;
        isp_config.de_config = &de;
    }
    if (config.ctc_table) {
        ctc_table = *config.ctc_table;
        isp_config.ctc_table = &ctc_table;
    }
    if (config.gc_config) {
        gc_config = *config.gc_config;
        isp_config.gc_config = &gc_config;
    }
    if (config.anr_config) {
        anr = *config.anr_config;
        isp_config.anr_config = &anr;
    }
    if (config.a3a_config) {
        a3a = *config.a3a_config;
        isp_config.a3a_config = &a3a;
    }
    if (config.xnr_config) {
        xnr = *config.xnr_config;
        isp_config.xnr_config = &xnr;
    }
    if (config.dz_config) {
        dz_config = *config.dz_config;
        isp_config.dz_config = &dz_config;
    }
    if (config.yuv2rgb_cc_config) {
        yuv2rgb_cc = *config.yuv2rgb_cc_config;
        isp_config.yuv2rgb_cc_config = &yuv2rgb_cc;
    }
    if (config.rgb2yuv_cc_config) {
        rgb2yuv_cc = *config.rgb2yuv_cc_config;
        isp_config.rgb2yuv_cc_config = &rgb2yuv_cc;
    }
    if (config.macc_table) {
        macc_table = *config.macc_table;
        isp_config.macc_table = &macc_table;
    }
    if (config.gamma_table) {
        gamma_table = *config.gamma_table;
        isp_config.gamma_table = &gamma_table;
    }
    if (config.r_gamma_table) {
        r_gamma_table = *config.r_gamma_table;
        isp_config.r_gamma_table = &r_gamma_table;
    }
    if (config.g_gamma_table) {
        g_gamma_table = *config.g_gamma_table;
        isp_config.g_gamma_table = &g_gamma_table;
    }
    if (config.b_gamma_table) {
        b_gamma_table = *config.b_gamma_table;
        isp_config.b_gamma_table = &b_gamma_table;
    }
    if (config.shading_table) {
        shading_table = *config.shading_table;
        isp_config.shading_table = &shading_table;
    }
    if (config.morph_table) {
        morph_table = *config.morph_table;
        isp_config.morph_table = &morph_table;
    }
    if (config.xnr_table) {
        xnr_table = *config.xnr_table;
        isp_config.xnr_table = &xnr_table;
    }
    if (config.anr_thres) {
        anr_thres = *config.anr_thres;
        isp_config.anr_thres = &anr_thres;
    }
    if (config.motion_vector) {
        motion_vector = *config.motion_vector;
        isp_config.motion_vector = &motion_vector;
    }
}

X3aIspConfig::X3aIspConfig ()
{
}

X3aIspConfig::~X3aIspConfig()
{
    clear ();
}


bool X3aIspConfig::clear()
{
    _isp_content.clear ();
    _3a_results.clear ();
    return true;
}

bool
X3aIspConfig::attach (SmartPtr<X3aResult> &result, IspConfigTranslator *translator)
{
    if (result.ptr() == NULL)
        return false;

    uint32_t type = result->get_type ();

    XCAM_ASSERT (translator);

    if (!result.ptr() || !result->get_ptr ()) {
        XCAM_LOG_ERROR ("3A result empty");
        return false;
    }
    switch (type) {
    case X3aIspConfig::IspAllParameters: {
        SmartPtr<X3aAtomIspParametersResult> isp_3a =
            result.dynamic_cast_ptr<X3aAtomIspParametersResult> ();
        XCAM_ASSERT (isp_3a.ptr ());
        _isp_content.copy (isp_3a->get_isp_config());
    }
    break;

    case XCAM_3A_RESULT_WHITE_BALANCE: {
        struct atomisp_wb_config wb;
        SmartPtr<X3aWhiteBalanceResult> wb_res =
            result.dynamic_cast_ptr<X3aWhiteBalanceResult> ();
        XCAM_ASSERT (wb_res.ptr ());
        xcam_mem_clear (wb);
        if (translator->translate_white_balance (wb_res->get_standard_result(), wb)
                != XCAM_RETURN_NO_ERROR) {
            XCAM_LOG_WARNING ("translate white balance failed");
            return false;
        }
        _isp_content.wb = wb;
        _isp_content.isp_config.wb_config = &_isp_content.wb;
    }
    break;
    case XCAM_3A_RESULT_BLACK_LEVEL: {
        struct atomisp_ob_config ob;
        SmartPtr<X3aBlackLevelResult> bl_res =
            result.dynamic_cast_ptr<X3aBlackLevelResult> ();
        XCAM_ASSERT (bl_res.ptr ());
        xcam_mem_clear (ob);
        if (translator->translate_black_level (bl_res->get_standard_result(), ob)
                != XCAM_RETURN_NO_ERROR) {
            XCAM_LOG_WARNING ("translate black level failed");
            return false;
        }
        _isp_content.ob = ob;
        _isp_content.isp_config.ob_config = &_isp_content.ob;
    }
    break;
    case XCAM_3A_RESULT_YUV2RGB_MATRIX:
    case XCAM_3A_RESULT_RGB2YUV_MATRIX:
    {
        struct atomisp_cc_config cc;
        SmartPtr<X3aColorMatrixResult> cc_res =
            result.dynamic_cast_ptr<X3aColorMatrixResult> ();
        XCAM_ASSERT (cc_res.ptr ());
        xcam_mem_clear (cc);
        if (translator->translate_color_matrix (cc_res->get_standard_result(), cc)
                != XCAM_RETURN_NO_ERROR) {
            XCAM_LOG_WARNING ("translate color matrix failed");
            return false;
        }
        if (type == XCAM_3A_RESULT_YUV2RGB_MATRIX) {
            _isp_content.yuv2rgb_cc = cc;
            _isp_content.isp_config.yuv2rgb_cc_config = &_isp_content.yuv2rgb_cc;
        } else {
            _isp_content.rgb2yuv_cc = cc;
            _isp_content.isp_config.rgb2yuv_cc_config = &_isp_content.rgb2yuv_cc;
        }
    }
    break;
    default:
        return false;
    }

    _3a_results.push_back (result);
    return true;
}

};

