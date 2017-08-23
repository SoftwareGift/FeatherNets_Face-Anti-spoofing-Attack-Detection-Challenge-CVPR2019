/*
 * context_priv.cpp - capi private context
 *
 *  Copyright (c) 2017 Intel Corporation
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

#include "context_priv.h"
#include <ocl/cl_device.h>
#include <ocl/cl_image_handler.h>
#include <ocl/cl_tonemapping_handler.h>
#include <ocl/cl_gauss_handler.h>
#include <ocl/cl_wavelet_denoise_handler.h>
#include <ocl/cl_newwavelet_denoise_handler.h>
#include <ocl/cl_defog_dcp_handler.h>
#include <ocl/cl_3d_denoise_handler.h>
#include <ocl/cl_image_warp_handler.h>
#include <ocl/cl_fisheye_handler.h>
#include <ocl/cl_image_360_stitch.h>

using namespace XCam;

#define DEFAULT_INPUT_BUFFER_POOL_COUNT  20
static const char *HandleNames[] = {
    "None",
    "3DNR",
    "WaveletNR",
    "Fisheye",
    "Defog",
    "DVS",
    "Stitch",
};

bool
handle_name_equal (const char *name, HandleType type)
{
    return !strncmp (name, HandleNames[type], strlen(HandleNames[type]));
}

ContextBase::ContextBase (HandleType type)
    : _type (type)
    , _usage (NULL)
    , _image_width (0)
    , _image_height (0)
    , _alloc_out_buf (false)
{
    if (!_inbuf_pool.ptr()) {
        SmartPtr<DrmDisplay> display = DrmDisplay::instance ();
        _inbuf_pool = new DrmBoBufferPool (display);
        XCAM_ASSERT (_inbuf_pool.ptr ());
    }
}

ContextBase::~ContextBase ()
{
    xcam_free (_usage);
}

const char*
ContextBase::get_type_name () const
{
    XCAM_ASSERT ((int)_type < sizeof(HandleNames) / sizeof (HandleNames[0]));
    return HandleNames [_type];
}

static const char*
find_value (const ContextParams &param_list, const char *name)
{
    ContextParams::const_iterator i = param_list.find (name);
    if (i != param_list.end ())
        return (i->second);
    return NULL;
}

XCamReturn
ContextBase::set_parameters (ContextParams &param_list)
{
    VideoBufferInfo buf_info;
    uint32_t image_format = V4L2_PIX_FMT_NV12;
    _image_width = 1920;
    _image_height = 1080;

    const char *width = find_value (param_list, "width");
    if (width) {
        _image_width = atoi(width);
    }
    const char *height = find_value (param_list, "height");
    if (height) {
        _image_height = atoi(height);
    }
    if (_image_width == 0 || _image_height == 0) {
        XCAM_LOG_ERROR ("illegal image size width:%d height:%d", _image_width, _image_height);
        return XCAM_RETURN_ERROR_PARAM;
    }

    buf_info.init (image_format, _image_width, _image_height);
    _inbuf_pool->set_video_info (buf_info);
    if (!_inbuf_pool->reserve (DEFAULT_INPUT_BUFFER_POOL_COUNT)) {
        XCAM_LOG_ERROR ("init buffer pool failed");
        return XCAM_RETURN_ERROR_MEM;
    }

    const char *flag = find_value (param_list, "alloc-out-buf");
    if (flag && !strncasecmp (flag, "true", strlen("true"))) {
        _alloc_out_buf = true;
    } else {
        _alloc_out_buf = false;
    }
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
ContextBase::init_handler ()
{
    SmartPtr<CLContext> cl_context = CLDevice::instance()->get_context ();
    XCAM_FAIL_RETURN (
        ERROR, cl_context.ptr (), XCAM_RETURN_ERROR_UNKNOWN,
        "ContextBase::init_handler(%s) failed since cl-context is NULL",
        get_type_name ());

    SmartPtr<CLImageHandler> handler = create_handler (cl_context);
    XCAM_FAIL_RETURN (
        ERROR, handler.ptr (), XCAM_RETURN_ERROR_UNKNOWN,
        "ContextBase::init_handler(%s) create handler failed", get_type_name ());

    handler->disable_buf_pool (!_alloc_out_buf);
    set_handler (handler);
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
ContextBase::uinit_handler ()
{
    if (!_handler.ptr ())
        return XCAM_RETURN_NO_ERROR;

    _handler->emit_stop ();
    _handler.release ();
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
ContextBase::execute (SmartPtr<DrmBoBuffer> &buf_in, SmartPtr<DrmBoBuffer> &buf_out)
{
    if (!_alloc_out_buf) {
        XCAM_FAIL_RETURN (
            ERROR, buf_out.ptr (), XCAM_RETURN_ERROR_MEM,
            "context (%s) execute failed, buf_out need set.", get_type_name ());
    } else {
        XCAM_FAIL_RETURN (
            ERROR, !buf_out.ptr (), XCAM_RETURN_ERROR_MEM,
            "context (%s) execute failed, buf_out need NULL.", get_type_name ());
    }

    return _handler->execute (buf_in, buf_out);
}

SmartPtr<CLImageHandler>
NR3DContext::create_handler (SmartPtr<CLContext> &context)
{
    return create_cl_3d_denoise_image_handler (
               context, CL_IMAGE_CHANNEL_Y | CL_IMAGE_CHANNEL_UV, 3);
}

SmartPtr<CLImageHandler>
NRWaveletContext::create_handler (SmartPtr<CLContext> &context)
{
    return create_cl_newwavelet_denoise_image_handler (
               context, CL_IMAGE_CHANNEL_UV | CL_IMAGE_CHANNEL_Y, false);
}

SmartPtr<CLImageHandler>
FisheyeContext::create_handler (SmartPtr<CLContext> &context)
{
    return create_fisheye_handler (context);
}

SmartPtr<CLImageHandler>
DefogContext::create_handler (SmartPtr<CLContext> &context)
{
    return create_cl_defog_dcp_image_handler (context);;
}

SmartPtr<CLImageHandler>
DVSContext::create_handler (SmartPtr<CLContext> &context)
{
    return create_cl_image_warp_handler (context);
}

SmartPtr<CLImageHandler>
StitchContext::create_handler (SmartPtr<CLContext> &context)
{
    uint32_t sttch_width = _image_width;
    uint32_t sttch_height = XCAM_ALIGN_UP (sttch_width / 2, 16);
    if (sttch_width != sttch_height * 2) {
        XCAM_LOG_ERROR ("incorrect stitch size width:%d height:%d", sttch_width, sttch_height);
        return NULL;
    }

    StitchResMode res_mode = StitchRes1080P;
    if (_res_mode == StitchRes4K)
        res_mode = StitchRes4K;

    SmartPtr<CLImage360Stitch> image_360 =
        create_image_360_stitch (context, _need_seam, _scale_mode,
                                 _fisheye_map, _need_lsc, res_mode).dynamic_cast_ptr<CLImage360Stitch> ();
    XCAM_FAIL_RETURN (ERROR, image_360.ptr (), NULL, "create image stitch handler failed");
    image_360->set_output_size (sttch_width, sttch_height);
    XCAM_LOG_INFO ("stitch output size width:%d height:%d", sttch_width, sttch_height);

#if HAVE_OPENCV
    image_360->set_feature_match_ocl (_fm_ocl);
#endif

    return image_360;
}

