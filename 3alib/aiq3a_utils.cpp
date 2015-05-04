/*
 * aiq3a_util.cpp - aiq 3a utility:
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
 * Author: Shincy Tu <shincy.tu@intel.com>
 */

#include "aiq3a_utils.h"
#include "x3a_isp_config.h"

using namespace XCam;

namespace XCamAiq3A {

bool
translate_3a_stats (XCam3AStats *from, struct atomisp_3a_statistics *to)
{
    XCAM_ASSERT (from);
    XCAM_ASSERT (to);

    struct atomisp_grid_info &to_info = to->grid_info;
    XCam3AStatsInfo &from_info = from->info;
    uint32_t color_count = (from_info.grid_pixel_size / 2) * (from_info.grid_pixel_size / 2);

    XCAM_ASSERT (to_info.bqs_per_grid_cell == 8);

    for (uint32_t i = 0; i < from_info.height; ++i)
        for (uint32_t j = 0; j < from_info.width; ++j) {
            to->data [i * to_info.aligned_width + j].ae_y =
                from->stats [i * from_info.aligned_width + j].avg_y * color_count;
            to->data [i * to_info.aligned_width + j].awb_gr =
                from->stats [i * from_info.aligned_width + j].avg_gr * color_count;
            to->data [i * to_info.aligned_width + j].awb_r =
                from->stats [i * from_info.aligned_width + j].avg_r * color_count;
            to->data [i * to_info.aligned_width + j].awb_b =
                from->stats [i * from_info.aligned_width + j].avg_b * color_count;
            to->data [i * to_info.aligned_width + j].awb_gb =
                from->stats [i * from_info.aligned_width + j].avg_gb * color_count;
            to->data [i * to_info.aligned_width + j].awb_cnt =
                from->stats [i * from_info.aligned_width + j].valid_wb_count;
            to->data [i * to_info.aligned_width + j].af_hpf1 =
                from->stats [i * from_info.aligned_width + j].f_value1;
            to->data [i * to_info.aligned_width + j].af_hpf2 =
                from->stats [i * from_info.aligned_width + j].f_value2;
        }
    return true;
}

uint32_t
translate_atomisp_parameters (
    const struct atomisp_parameters &atomisp_params,
    XCam3aResultHead *results[], uint32_t max_count)
{
    uint32_t result_count = 0;

    return result_count;
}

uint32_t
translate_3a_results_to_xcam (X3aResultList &list,
                              XCam3aResultHead *results[], uint32_t max_count)
{
    uint32_t result_count = 0;
    for (X3aResultList::iterator iter = list.begin (); iter != list.end (); ++iter) {
        SmartPtr<X3aResult> &isp_result = *iter;

        switch (isp_result->get_type()) {
        case X3aIspConfig::IspExposureParameters: {
            SmartPtr<X3aIspExposureResult> isp_exposure =
                isp_result.dynamic_cast_ptr<X3aIspExposureResult> ();
            XCAM_ASSERT (isp_exposure.ptr ());
            const XCam3aResultExposure &exposure = isp_exposure->get_standard_result ();
            XCam3aResultExposure *new_exposure = xcam_malloc0_type (XCam3aResultExposure);
            *new_exposure = exposure;
            results[result_count++] = (XCam3aResultHead*)new_exposure;
            break;
        }
        case X3aIspConfig::IspAllParameters: {
            SmartPtr<X3aAtomIspParametersResult> isp_3a_all =
                isp_result.dynamic_cast_ptr<X3aAtomIspParametersResult> ();
            XCAM_ASSERT (isp_3a_all.ptr ());
            const struct atomisp_parameters &atomisp_params = isp_3a_all->get_isp_config ();
            translate_atomisp_parameters (atomisp_params, &results[result_count], max_count - result_count);
            break;
        }
        default:
            XCAM_LOG_WARNING ("unknow type(%d) in translation", isp_result->get_type());
            break;
        }
    }
    return result_count;
}

void
free_3a_result (XCam3aResultHead *result)
{
    xcam_free (result);
}

}
