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

#ifndef XCAM_3A_ISP_CONFIG_H
#define XCAM_3A_ISP_CONFIG_H

#include "xcam_utils.h"
#include "x3a_result.h"
#include <linux/atomisp.h>
#include <base/xcam_3a_result.h>

namespace XCam {

#define XCAM_3A_ISP_RESULT_TYPE_START (XCAM_3A_RESULT_USER_DEFINED_TYPE + 0x1000)

struct AtomIspConfigContent {
    struct atomisp_parameters   isp_config;
    //content
    struct atomisp_wb_config     wb;
    struct atomisp_ob_config     ob; //black level
    struct atomisp_cc_config     cc;
    struct atomisp_cc_config     yuv2rgb_cc;
    struct atomisp_cc_config     rgb2yuv_cc;
    struct atomisp_nr_config     nr;
    struct atomisp_tnr_config    tnr;
    struct atomisp_ynr_config    ynr;
    struct atomisp_cnr_config    cnr;
    struct atomisp_anr_config    anr;
    struct atomisp_xnr_config    xnr;
    struct atomisp_xnr_table     xnr_table;
    struct atomisp_ee_config     ee;
    struct atomisp_dp_config     dp;
    struct atomisp_de_config     de;
    struct atomisp_ecd_config    ecd_config;
    struct atomisp_fc_config     fc_config;
    struct atomisp_ctc_config    ctc_config;
    struct atomisp_ctc_table     ctc_table;
    struct atomisp_macc_config   macc_config;
    struct atomisp_macc_table    macc_table;
    struct atomisp_gamma_table   gamma_table;
    struct atomisp_rgb_gamma_table r_gamma_table;
    struct atomisp_rgb_gamma_table g_gamma_table;
    struct atomisp_rgb_gamma_table b_gamma_table;
    struct atomisp_gc_config     gc_config;
    struct atomisp_shading_table shading_table;
    struct atomisp_3a_config     a3a;

    struct atomisp_dvs_6axis_config dvs_6axis;


    struct atomisp_formats_config formats;
    struct atomisp_aa_config     aa;
    struct atomisp_aa_config     baa;
    struct atomisp_ce_config     ce;
    struct atomisp_morph_table   morph_table;
    struct atomisp_anr_thres     anr_thres;

    struct atomisp_dz_config     dz_config;
    struct atomisp_vector        motion_vector;

    void clear ();
    void copy (const struct atomisp_parameters &config);

    AtomIspConfigContent () {
        clear ();
    }
};

class IspConfigTranslator;

class X3aIspConfig
{
public:
    enum X3aIspConfigType {
        IspAllParameters = XCAM_3A_ISP_RESULT_TYPE_START,
        IspExposureParameters,
    };

    struct X3aIspResultDummy {
        XCam3aResultHead head;
    };
public:
    explicit X3aIspConfig ();
    virtual ~X3aIspConfig();

public:
    const struct atomisp_parameters &get_isp_configs () const {
        return _isp_content.isp_config;
    }
    struct atomisp_parameters &get_isp_configs () {
        return _isp_content.isp_config;
    }
    bool clear ();
    bool attach (SmartPtr<X3aResult> &result, IspConfigTranslator *translator);

private:
    XCAM_DEAD_COPY (X3aIspConfig);

protected:
    AtomIspConfigContent             _isp_content;
    std::list< SmartPtr<X3aResult> > _3a_results;
};

template <typename IspConfig, typename StandardResult, uint32_t type>
class X3aIspResultT
    : public X3aStandardResultT<StandardResult>
{
public:
    X3aIspResultT (
        XCamImageProcessType process_type = XCAM_IMAGE_PROCESS_ALWAYS
    )
        : X3aStandardResultT<StandardResult> (type, process_type)
    {
        X3aResult::set_ptr((void*)&_isp_config);
    }

    ~X3aIspResultT () {}

    // set config
    void set_isp_config (IspConfig &config) {
        _isp_config = config;
    }
    const IspConfig &get_isp_config () const {
        return _isp_config;
    }

private:
    IspConfig _isp_config;
};


/* special X3aAtomIspParametersResult type */
template <>
class X3aIspResultT<struct atomisp_parameters, X3aIspConfig::X3aIspResultDummy, X3aIspConfig::IspAllParameters>
            : public X3aStandardResultT<X3aIspConfig::X3aIspResultDummy>
    {
public:
        X3aIspResultT (
            XCamImageProcessType process_type = XCAM_IMAGE_PROCESS_ALWAYS)
            : X3aStandardResultT<X3aIspConfig::X3aIspResultDummy> ((uint32_t)X3aIspConfig::IspAllParameters, process_type)
        {
            X3aResult::set_ptr((void*)&_content.isp_config);
        }

        ~X3aIspResultT () {}

        // get config
        struct atomisp_parameters &get_isp_config () {
            return _content.isp_config;
        }
        const struct atomisp_parameters &get_isp_config () const {
            return _content.isp_config;
        }

        // set config
        void set_isp_config (struct atomisp_parameters &config) {
            _content.copy (config);
        }

private:
        AtomIspConfigContent      _content;
    };

typedef
X3aIspResultT<struct atomisp_parameters, X3aIspConfig::X3aIspResultDummy, X3aIspConfig::IspAllParameters> X3aAtomIspParametersResult;
typedef
X3aIspResultT<struct atomisp_exposure, XCam3aResultExposure, X3aIspConfig::IspExposureParameters> X3aIspExposureResult;

};

#endif //XCAM_3A_ISP_CONFIG_H

