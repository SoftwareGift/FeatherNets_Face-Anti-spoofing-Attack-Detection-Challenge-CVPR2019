/*
 * xcam_smart_description.h - libxcam smart analysis description
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
 * Author: Zong Wei <wei.zong@intel.com>
 *         Wind Yuan <feng.yuan@intel.com>
 */

#ifndef C_XCAM_SMART_ANALYSIS_DESCRIPTION_H
#define C_XCAM_SMART_ANALYSIS_DESCRIPTION_H

#include <base/xcam_common.h>
#include <base/xcam_params.h>
#include <base/xcam_3a_result.h>
#include <base/xcam_buffer.h>

XCAM_BEGIN_DECLARE

#define XCAM_SMART_ANALYSIS_LIB_DESCRIPTION "xcam_smart_analysis_desciption"

typedef struct _XCamSmartAnalysisContext XCamSmartAnalysisContext;

typedef XCamReturn (*XcamPostResultsFunc) (
    XCamSmartAnalysisContext *context,
    const XCamVideoBuffer *buffer,
    XCam3aResultHead *results[], uint32_t res_count);


#define XCAM_SMART_PLUGIN_PRIORITY_HIGH     1
#define XCAM_SMART_PLUGIN_PRIORITY_DEFAULT  10
#define XCAM_SMART_PLUGIN_PRIORITY_LOW      100

/*  \brief C interface of Smart Analysis Description
  *     <version>          xcam version
  *     <size>               description structure size, sizeof (XCamSmartAnalysisDescription)
  *     <priority>           smart plugin priority; the less value the higher priority; 0, highest priority
  *     <name>             smart pluign name, or use file name if NULL
  */
typedef struct _XCamSmartAnalysisDescription {
    uint32_t                        version;
    uint32_t                        size;
    uint32_t                        priority;
    const char                     *name;

    /*! \brief initialize smart analysis context.
    *
    * \param[out]        context            create context handle
    * \param[out]        async_mode     0, sync mode; 1, async mode
    * \param[in]          post_func         plugin can use post_func to post results in async mode
    */
    XCamReturn (*create_context)  (XCamSmartAnalysisContext **context,
                                   uint32_t *async_mode, XcamPostResultsFunc post_func);
    /*! \brief destroy smart analysis context.
    *
    * \param[in]        context            create context handle
    */
    XCamReturn (*destroy_context) (XCamSmartAnalysisContext *context);

    /*! \brief update smart analysis context parameters.
    *
    * \param[in]        context            context handle
    * \param[in]        params            new parameters
    */
    XCamReturn (*update_params)   (XCamSmartAnalysisContext *context, const XCamSmartAnalysisParam *params);

    /*! \brief analyze data and get result,.
    *
    * \param[in]        context            context handle
    * \param[in]        buffer              image buffer
    * \param[out]      results             analysis results array, only for sync mode (<async_mode> = 0)
    * \param[in/out]  res_count        in, max results array size; out, return results count.
    */
    XCamReturn (*analyze)         (XCamSmartAnalysisContext *context, XCamVideoBuffer *buffer,
                                   XCam3aResultHead *results[], uint32_t *res_count);

    /*! \brief free smart results.
    *
    * \param[in]        context            context handle
    * \param[in]        results             analysis results
    * \param[in]        res_count         analysis results count
    */
    void       (*free_results)    (XCamSmartAnalysisContext *context, XCam3aResultHead *results[], uint32_t res_count);
} XCamSmartAnalysisDescription;

XCAM_END_DECLARE

#endif //C_XCAM_SMART_ANALYSIS_DESCRIPTION_H
