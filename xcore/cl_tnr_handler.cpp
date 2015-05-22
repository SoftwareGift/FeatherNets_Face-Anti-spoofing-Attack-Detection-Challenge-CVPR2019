/*
 * cl_tnr_handler.cpp - CL tnr handler
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
 * Author: Wei Zong <wei.zong@intel.com>
 */
#include "xcam_utils.h"
#include "cl_tnr_handler.h"

namespace XCam {

CLTnrImageKernel::CLTnrImageKernel (SmartPtr<CLContext> &context,
                                    const char *name,
                                    CLTnrType type)
    : CLImageKernel (context, name, false)
    , _type (type)
    , _gain (0.5)
    , _thr_Y (0.05)
    , _thr_C (0.05)
    , _frame_count (TNR_PROCESSING_FRAME_COUNT)
{
}

XCamReturn
CLTnrImageKernel::prepare_arguments (
    SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
    CLArgument args[], uint32_t &arg_count,
    CLWorkSize &work_size)
{
    SmartPtr<CLContext> context = get_context ();

    const VideoBufferInfo & video_info = input->get_video_info ();

    _image_in = new CLVaImage (context, input);
    if (CL_TNR_TYPE_RGB == _type) {
        if (_image_in_list.size () < _frame_count) {
            while (_image_in_list.size () < _frame_count) {
                _image_in_list.push_back (_image_in);
            }
        } else {
            _image_in_list.pop_front ();
            _image_in_list.push_back (_image_in);
        }
    }

    _image_out = new CLVaImage (context, output);

    if (CL_TNR_TYPE_YUV == _type) {
        if (!_image_out_prev.ptr ()) {
            _image_out_prev = _image_in;
        }
    }
    _vertical_offset = video_info.aligned_height;

    XCAM_ASSERT (_image_in->is_valid () && _image_out->is_valid ());
    XCAM_FAIL_RETURN (
        WARNING,
        _image_in->is_valid () && _image_out->is_valid (),
        XCAM_RETURN_ERROR_MEM,
        "cl image kernel(%s) in/out memory not available", get_kernel_name ());

    //set args;
    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    work_size.local[0] = 8;
    work_size.local[1] = 4;
    if (CL_TNR_TYPE_YUV == _type) {
        args[0].arg_adress = &_image_in->get_mem_id ();
        args[0].arg_size = sizeof (cl_mem);

        args[1].arg_adress = &_image_out_prev->get_mem_id ();
        args[1].arg_size = sizeof (cl_mem);

        args[2].arg_adress = &_image_out->get_mem_id ();
        args[2].arg_size = sizeof (cl_mem);

        args[3].arg_adress = &_vertical_offset;
        args[3].arg_size = sizeof (_vertical_offset);

        args[4].arg_adress = &_gain;
        args[4].arg_size = sizeof (_gain);

        args[5].arg_adress = &_thr_Y;
        args[5].arg_size = sizeof (_thr_Y);

        args[6].arg_adress = &_thr_C;
        args[6].arg_size = sizeof (_thr_C);

        work_size.global[0] = video_info.width / 2;
        work_size.global[1] = video_info.height / 2;
        arg_count = 7;
    }
    else if (CL_TNR_TYPE_RGB == _type) {
        const CLImageDesc out_info = _image_out->get_image_desc ();
        work_size.global[0] = out_info.width;
        work_size.global[1] = out_info.height;

        args[0].arg_adress = &_image_out->get_mem_id ();
        args[0].arg_size = sizeof (cl_mem);

        args[1].arg_adress = &_thr_Y;
        args[1].arg_size = sizeof (_thr_Y);

        args[2].arg_adress = &_frame_count;
        args[2].arg_size = sizeof (_frame_count);

        uint8_t index = 0;
        for (std::list<SmartPtr<CLImage>>::iterator it = _image_in_list.begin (); it != _image_in_list.end (); it++) {
            args[3 + index].arg_adress = &(*it)->get_mem_id ();
            args[3 + index].arg_size = sizeof (cl_mem);
            index++;
        }

        arg_count = 3 + index;
    }

    return XCAM_RETURN_NO_ERROR;
}


XCamReturn
CLTnrImageKernel::post_execute ()
{
    if ((CL_TNR_TYPE_YUV == _type) && _image_out->is_valid ()) {
        _image_out_prev = _image_out;
    }

    return CLImageKernel::post_execute ();
}

bool
CLTnrImageKernel::set_gain (float gain)
{
    XCAM_LOG_DEBUG ("set TNR gain(%f)", gain);

    _gain = gain;
    return true;
}

bool
CLTnrImageKernel::set_threshold (float thr_y, float thr_uv)
{
    XCAM_LOG_DEBUG ("set TNR threshold: Y(%f), UV(%f)", thr_y, thr_uv);

    _thr_Y = thr_y;
    _thr_C = thr_uv;

    return true;
}

CLTnrImageHandler::CLTnrImageHandler (const char *name)
    : CLImageHandler (name)
{
}

bool
CLTnrImageHandler::set_tnr_kernel(SmartPtr<CLTnrImageKernel> &kernel)
{
    SmartPtr<CLImageKernel> image_kernel = kernel;
    add_kernel (image_kernel);
    _tnr_kernel = kernel;
    return true;
}

bool
CLTnrImageHandler::set_mode (uint32_t mode)
{
    if (!_tnr_kernel->is_valid ()) {
        XCAM_LOG_ERROR ("set mode error, invalid TNR kernel !");
    }

    _tnr_kernel->set_enable (mode & (CL_TNR_TYPE_YUV | CL_TNR_TYPE_RGB));
    return true;
}

bool
CLTnrImageHandler::set_gain (float gain)
{
    if (!_tnr_kernel->is_valid ()) {
        XCAM_LOG_ERROR ("set gain error, invalid TNR kernel !");
    }

    _tnr_kernel->set_gain (gain);

    return true;
}

bool
CLTnrImageHandler::set_threshold (float thr_y, float thr_uv)
{
    if (!_tnr_kernel->is_valid ()) {
        XCAM_LOG_ERROR ("set threshold error, invalid TNR kernel !");
    }

    _tnr_kernel->set_threshold (thr_y, thr_uv);

    return true;
}

SmartPtr<CLImageHandler>
create_cl_tnr_image_handler (SmartPtr<CLContext> &context, CLTnrType type)
{
    SmartPtr<CLTnrImageHandler> tnr_handler;
    SmartPtr<CLTnrImageKernel> tnr_kernel;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    XCAM_CL_KERNEL_FUNC_SOURCE_BEGIN(kernel_tnr_yuv)
#include "kernel_tnr_yuv.clx"
    XCAM_CL_KERNEL_FUNC_END;

    XCAM_CL_KERNEL_FUNC_SOURCE_BEGIN(kernel_tnr_rgb)
#include "kernel_tnr_rgb.clx"
    XCAM_CL_KERNEL_FUNC_END;

    if (CL_TNR_TYPE_YUV == type) {
        tnr_kernel = new CLTnrImageKernel (context, "kernel_tnr_yuv", CL_TNR_TYPE_YUV);
        ret = tnr_kernel->load_from_source (kernel_tnr_yuv_body, strlen (kernel_tnr_yuv_body));
    } else if (CL_TNR_TYPE_RGB == type) {
        tnr_kernel = new CLTnrImageKernel (context, "kernel_tnr_rgb", CL_TNR_TYPE_RGB);
        ret = tnr_kernel->load_from_source (kernel_tnr_rgb_body, strlen (kernel_tnr_rgb_body));
    }

    XCAM_FAIL_RETURN (
        WARNING,
        ret == XCAM_RETURN_NO_ERROR,
        NULL,
        "CL image handler(%s) load source failed", tnr_kernel->get_kernel_name());

    tnr_handler = new CLTnrImageHandler ("cl_handler_tnr");
    XCAM_ASSERT (tnr_kernel->is_valid ());
    tnr_handler->set_tnr_kernel (tnr_kernel);

    return tnr_handler;
}

};
