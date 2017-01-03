/*
 * xcam_plugin_dvs.cpp - Digital Video Stabilizer plugin
 *
 *  Copyright (c) 2014-2016 Intel Corporation
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
#include <base/xcam_common.h>
#include <base/xcam_smart_description.h>
#include <base/xcam_smart_result.h>
#include <base/xcam_3a_result.h>
#include <base/xcam_buffer.h>

#include <xcam_utils.h>
#include <smartptr.h>
#include <drm_display.h>
#include <dma_video_buffer.h>

#include <cl_context.h>
#include <cl_device.h>
#include <cl_memory.h>

#include <opencv2/core/ocl.hpp>

#include "libdvs/libdvs.h"

#define DVS_MOTION_FILTER_RADIUS   15

struct DvsBuffer : public DvsData
{
    XCamVideoBuffer* buffer;

    DvsBuffer () { }

    DvsBuffer (XCamVideoBuffer* buf, cv::UMat& frame)
        : buffer (buf)
    {
        buffer->ref(buffer);
        data = frame;
    }

    ~DvsBuffer () {
        buffer->unref(buffer);
    }
};

XCamReturn dvs_create_context(XCamSmartAnalysisContext **context, uint32_t *async_mode, XcamPostResultsFunc post_func)
{
    XCAM_UNUSED (async_mode);
    XCAM_UNUSED (post_func);

    DvsInterface* theDVS = NULL;

    theDVS = getDigitalVideoStabilizer();
    if (theDVS == NULL) {
        return XCAM_RETURN_ERROR_MEM;
    }
    theDVS->init(640, 480, false);

    *context = (XCamSmartAnalysisContext *)theDVS;

    cl_platform_id platform_id = XCam::CLDevice::instance()->get_platform_id();
    char* platform_name = XCam::CLDevice::instance()->get_platform_name ();
    cl_device_id device_id = XCam::CLDevice::instance()->get_device_id();
    cl_context cl_context_id = XCam::CLDevice::instance()->get_context()->get_context_id();

    clRetainContext (cl_context_id);
    cv::ocl::attachContext (platform_name, platform_id, cl_context_id, device_id);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn dvs_destroy_context(XCamSmartAnalysisContext *context)
{
    DvsInterface *theDVS = (DvsInterface *)context;

    theDVS->release ();

    delete (theDVS);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn dvs_update_params(XCamSmartAnalysisContext *context, const XCamSmartAnalysisParam *params)
{
    XCAM_UNUSED (context);
    XCAM_UNUSED (params);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn dvs_analyze(XCamSmartAnalysisContext *context, XCamVideoBuffer *buffer, XCam3aResultHead *results[], uint32_t *res_count)
{
    DvsInterface *theDVS = (DvsInterface *)context;
    DvsResult dvsResult;

    if (buffer->info.format != V4L2_PIX_FMT_NV12 || buffer->mem_type != XCAM_MEM_TYPE_PRIVATE_BO)
        return XCAM_RETURN_ERROR_PARAM;

    int buffer_fd = xcam_video_buffer_get_fd(buffer);
    XCam::VideoBufferInfo buffer_info;
    buffer_info.init (buffer->info.format, buffer->info.width, buffer->info.height,
                      buffer->info.aligned_width, buffer->info.aligned_height, buffer->info.size);
    XCam::SmartPtr<XCam::VideoBuffer> video_buffer = new XCam::DmaVideoBuffer(buffer_info, buffer_fd);

    XCam::SmartPtr<XCam::DrmDisplay> display = XCam::DrmDisplay::instance ();
    XCam::SmartPtr<XCam::DrmBoBuffer> bo_buffer = display->convert_to_drm_bo_buf (display, video_buffer);

    XCam::SmartPtr<XCam::CLContext> cl_Context = XCam::CLDevice::instance()->get_context();
    XCam::SmartPtr<XCam::CLBuffer> cl_buffer = new XCam::CLVaBuffer (cl_Context, bo_buffer);
    cl_mem cl_mem_id = cl_buffer->get_mem_id();

    clRetainMemObject(cl_mem_id);
    cv::UMat frame;
    cv::ocl::convertFromBuffer(cl_mem_id, buffer->info.strides[0], buffer->info.height, buffer->info.width, CV_8U, frame);

    DvsBuffer* dvs_buf = new DvsBuffer(buffer, frame);
    //set default config
    DvsConfig config;
    memset(&config, 0, sizeof(DvsConfig));
    config.use_ocl  = true;
    config.frame_width = buffer->info.width;
    config.frame_height = buffer->info.height;
    config.radius = DVS_MOTION_FILTER_RADIUS;
    config.stdev = 10.0f;
    config.features = 1000;
    config.minDistance = 20.0f;

    theDVS->setConfig(&config);

    theDVS->nextStabilizedMotion(dvs_buf, &dvsResult);

    delete(dvs_buf);

    if ((dvsResult.frame_id < 0) && (dvsResult.valid == false))
    {
        results[0] = NULL;
        *res_count = 0;
        XCAM_LOG_WARNING ("dvs_analyze not ready! ");
    } else {
        XCamDVSResult *dvs_result = (XCamDVSResult *)malloc(sizeof(XCamDVSResult));
        memset(dvs_result, 0, sizeof(XCamDVSResult));

        dvs_result->head.type = XCAM_3A_RESULT_DVS;
        dvs_result->head.process_type = XCAM_IMAGE_PROCESS_POST;
        dvs_result->head.version = 0x080;
        dvs_result->frame_id = dvsResult.frame_id;
        dvs_result->frame_width = dvsResult.frame_width;
        dvs_result->frame_height = dvsResult.frame_height;
        memcpy(dvs_result->proj_mat, dvsResult.proj_mat, sizeof(DvsResult::proj_mat));

        results[0] = (XCam3aResultHead *)dvs_result;
        *res_count = 1;
    }

    return XCAM_RETURN_NO_ERROR;
}

void dvs_free_results(XCamSmartAnalysisContext *context, XCam3aResultHead *results[], uint32_t res_count)
{
    XCAM_UNUSED (context);
    for (uint32_t i = 0; i < res_count; ++i) {
        if (results[i]) {
            free (results[i]);
        }
    }
}

XCAM_BEGIN_DECLARE

XCamSmartAnalysisDescription xcam_smart_analysis_desciption =
{
    0x080,
    sizeof (XCamSmartAnalysisDescription),
    10,
    "digital_video_stabilizer",
    dvs_create_context,
    dvs_destroy_context,
    dvs_update_params,
    dvs_analyze,
    dvs_free_results,
};

XCAM_END_DECLARE

