/*
 * x3a_image_process_center.cpp - 3a process center
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
#include "x3a_image_process_center.h"

namespace XCam {

X3aImageProcessCenter::X3aImageProcessCenter()
    :   _callback (NULL)
{
    XCAM_LOG_DEBUG ("X3aImageProcessCenter construction");
}

X3aImageProcessCenter::~X3aImageProcessCenter()
{
    stop ();
    XCAM_LOG_DEBUG ("~X3aImageProcessCenter destruction");
}

bool
X3aImageProcessCenter::set_image_callback (ImageProcessCallback *callback)
{
    XCAM_ASSERT (!_callback);
    _callback = callback;
    return true;
}

bool
X3aImageProcessCenter::insert_processor (SmartPtr<ImageProcessor> &processor)
{
    _image_processors.push_back (processor);
    XCAM_LOG_INFO ("Add processor(%s) into image processor center", XCAM_STR (processor->get_name()));
    return true;
}

bool
X3aImageProcessCenter::has_processors ()
{
    return !_image_processors.empty();
}

XCamReturn
X3aImageProcessCenter::start ()
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    if (_image_processors.empty()) {
        XCAM_LOG_ERROR ("process center start failed, no processor found");
        return XCAM_RETURN_ERROR_PARAM;
    }

    for (ImageProcessorList::iterator i_pro = _image_processors.begin ();
            i_pro != _image_processors.end(); ++i_pro)
    {
        SmartPtr<ImageProcessor> &processor = *i_pro;
        XCAM_ASSERT (processor.ptr());
        processor->set_callback (this);
        ret = processor->start ();
        if (ret != XCAM_RETURN_NO_ERROR) {
            XCAM_LOG_ERROR ("processor(%s) start failed", XCAM_STR(processor->get_name()));
            break;
        }
    }

    if (ret != XCAM_RETURN_NO_ERROR)
        stop();
    else {
        XCAM_LOG_INFO ("3a process center started");
    }

    return ret;
}

XCamReturn
X3aImageProcessCenter::stop ()
{
    for (ImageProcessorList::iterator i_pro = _image_processors.begin ();
            i_pro != _image_processors.end(); ++i_pro)
    {
        SmartPtr<ImageProcessor> &processor = *i_pro;
        XCAM_ASSERT (processor.ptr());
        processor->stop ();
    }

    XCAM_LOG_INFO ("3a process center stopped");

    _image_processors.clear();
    return XCAM_RETURN_NO_ERROR;
}

bool
X3aImageProcessCenter::put_buffer (SmartPtr<VideoBuffer> &buf)
{
    XCAM_ASSERT (!_image_processors.empty());
    if (_image_processors.empty())
        return false;

    ImageProcessorList::iterator i_pro = _image_processors.begin ();
    SmartPtr<ImageProcessor> &processor = *i_pro;
    if (processor->push_buffer (buf) != XCAM_RETURN_NO_ERROR)
        return false;
    return true;
}


XCamReturn
X3aImageProcessCenter::put_3a_results (X3aResultList &results)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    XCAM_FAIL_RETURN (ERROR, !results.empty(), XCAM_RETURN_ERROR_PARAM, "results empty");

    for (ImageProcessorIter i_pro = _image_processors.begin();
            i_pro != _image_processors.end(); i_pro++) {
        SmartPtr<ImageProcessor> &processor = *i_pro;
        XCAM_ASSERT (processor.ptr());
        ret = processor->push_3a_results (results);
        if (ret != XCAM_RETURN_NO_ERROR && ret != XCAM_RETURN_BYPASS) {
            XCAM_LOG_WARNING ("processor(%s) gailed on results", XCAM_STR(processor->get_name()));
            break;
        }
        if (results.empty ()) {
            XCAM_LOG_DEBUG ("results done");
            return XCAM_RETURN_NO_ERROR;
        }
    }

    if (!results.empty()) {
        XCAM_LOG_DEBUG ("process center: results left without being processed");
        return XCAM_RETURN_BYPASS;
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
X3aImageProcessCenter::put_3a_result (SmartPtr<X3aResult> &result)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    XCAM_FAIL_RETURN (ERROR, !result.ptr(), XCAM_RETURN_ERROR_PARAM, "result empty");

    for (ImageProcessorIter i_pro = _image_processors.begin();
            i_pro != _image_processors.end(); i_pro++)
    {
        SmartPtr<ImageProcessor> &processor = *i_pro;
        XCAM_ASSERT (processor.ptr());
        ret = processor->push_3a_result (result);

        if (ret == XCAM_RETURN_BYPASS)
            continue;

        if (ret == XCAM_RETURN_NO_ERROR)
            return XCAM_RETURN_NO_ERROR;

        XCAM_LOG_WARNING ("processor(%s) failed on result", XCAM_STR(processor->get_name()));
        return ret;
    }

    if (ret == XCAM_RETURN_BYPASS) {
        XCAM_LOG_WARNING ("processor center: no processor can handle result()");
    }

    return ret;
}

void
X3aImageProcessCenter::process_buffer_done (ImageProcessor *processor, const SmartPtr<VideoBuffer> &buf)
{
    ImageProcessorIter i_pro = _image_processors.begin();
    for (; i_pro != _image_processors.end(); ++i_pro)
    {
        SmartPtr<ImageProcessor> &cur_pro = *i_pro;
        XCAM_ASSERT (cur_pro.ptr());
        if (cur_pro.ptr() == processor)
            break;
    }

    XCAM_ASSERT (i_pro != _image_processors.end());
    if (i_pro == _image_processors.end()) {
        XCAM_LOG_ERROR ("processor doesn't found from list of image center");
        return;
    }

    if (++i_pro != _image_processors.end()) {
        SmartPtr<ImageProcessor> &next_processor = *i_pro;
        SmartPtr<VideoBuffer> cur_buf = buf;
        XCAM_ASSERT (next_processor.ptr());
        XCamReturn ret = next_processor->push_buffer (cur_buf);
        if (ret != XCAM_RETURN_NO_ERROR) {
            XCAM_LOG_ERROR ("processor(%s) failed in push_buffer", next_processor->get_name());
        }
        return;
    }

    //all processor done
    if (_callback)
        _callback->process_buffer_done (processor, buf);
    else
        ImageProcessCallback::process_buffer_done (processor, buf);
}

void
X3aImageProcessCenter::process_buffer_failed (ImageProcessor *processor, const SmartPtr<VideoBuffer> &buf)
{
    if (_callback)
        _callback->process_buffer_failed(processor, buf);
    else
        ImageProcessCallback::process_buffer_failed (processor, buf);
}

void
X3aImageProcessCenter::process_image_result_done (ImageProcessor *processor, const SmartPtr<X3aResult> &result)
{
    if (_callback)
        _callback->process_image_result_done(processor, result);
    else
        ImageProcessCallback::process_image_result_done (processor, result);
}

};
