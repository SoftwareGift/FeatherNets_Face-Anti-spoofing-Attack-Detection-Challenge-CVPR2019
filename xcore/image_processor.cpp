/*
 * image_processor.h - 3a image processor
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

#include "image_processor.h"
#include "xcam_thread.h"

namespace XCam {

void
ImageProcessCallback::process_buffer_done (ImageProcessor *processor, SmartPtr<VideoBuffer> &buf) {
    XCAM_ASSERT (buf.ptr() && processor);

    int64_t ts = buf->get_timestamp();
    XCAM_UNUSED (ts);
    XCAM_LOG_DEBUG (
        "processor(%s) handled buffer(" XCAM_TIMESTAMP_FORMAT ") successfully",
        XCAM_STR(processor->get_name()),
        XCAM_TIMESTAMP_ARGS (ts));
}

void
ImageProcessCallback::process_buffer_failed (ImageProcessor *processor, SmartPtr<VideoBuffer> &buf)
{
    XCAM_ASSERT (buf.ptr() && processor);

    int64_t ts = buf->get_timestamp();
    XCAM_UNUSED (ts);
    XCAM_LOG_WARNING (
        "processor(%s) handled buffer(" XCAM_TIMESTAMP_FORMAT ") failed",
        XCAM_STR(processor->get_name()),
        XCAM_TIMESTAMP_ARGS (ts));
}

void
ImageProcessCallback::process_image_result_done (ImageProcessor *processor, SmartPtr<X3aResult> &result)
{
    XCAM_ASSERT (result.ptr() && processor);

    int64_t ts = result->get_timestamp();
    XCAM_UNUSED (ts);

    XCAM_LOG_DEBUG (
        "processor(%s) processed result(type:%d, timestamp:" XCAM_TIMESTAMP_FORMAT ") done",
        XCAM_STR(processor->get_name()),
        (int)result->get_type(),
        XCAM_TIMESTAMP_ARGS (ts));
}

class ImageProcessorThread
    : public Thread
{
public:
    ImageProcessorThread (ImageProcessor *processor)
        : Thread ("image_processor")
        , _processor (processor)
    {}
    ~ImageProcessorThread () {}

    virtual bool loop ();

private:
    ImageProcessor *_processor;
};

bool ImageProcessorThread::loop ()
{
    XCamReturn ret = _processor->buffer_process_loop ();
    if (ret == XCAM_RETURN_NO_ERROR || ret == XCAM_RETURN_ERROR_TIMEOUT)
        return true;
    return false;
}

ImageProcessor::ImageProcessor (const char* name)
    : _name (NULL)
    , _callback (NULL)
{
    _processor_thread = new ImageProcessorThread (this);
    if (name)
        _name = strdup (name);
}

ImageProcessor::~ImageProcessor ()
{
    if (_name)
        xcam_free (_name);
}

bool
ImageProcessor::set_callback (ImageProcessCallback *callback)
{
    XCAM_ASSERT (!_callback);
    _callback = callback;
    return true;
}

XCamReturn
ImageProcessor::start()
{
    if (!_processor_thread->start()) {
        return XCAM_RETURN_ERROR_THREAD;
    }
    XCAM_LOG_INFO ("ImageProcessor(%s) started", XCAM_STR (_name));
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
ImageProcessor::stop()
{
    _video_buf_queue.wakeup ();

    emit_stop ();

    _processor_thread->stop ();
    XCAM_LOG_DEBUG ("ImageProcessor(%s) stopped", XCAM_STR (_name));
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
ImageProcessor::push_buffer (SmartPtr<VideoBuffer> &buf)
{
    if (_video_buf_queue.push (buf))
        return XCAM_RETURN_NO_ERROR;

    XCAM_LOG_DEBUG ("processor push buffer failed");
    return XCAM_RETURN_ERROR_UNKNOWN;
}

XCamReturn
ImageProcessor::push_3a_results (X3aResultList &results)
{
    X3aResultList valid_results;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    filter_valid_results (results, valid_results);
    if (valid_results.empty())
        return XCAM_RETURN_BYPASS;

    ret = apply_3a_results (valid_results);

    if (ret != XCAM_RETURN_NO_ERROR && ret != XCAM_RETURN_BYPASS) {
        XCAM_LOG_WARNING ("processor(%s) apply results failed", XCAM_STR(get_name()));
        return ret;
    }

    if (_callback) {
        for (X3aResultList::iterator i_res = valid_results.begin();
                i_res != valid_results.end(); ++i_res) {
            SmartPtr<X3aResult> &res = *i_res;
            _callback->process_image_result_done (this, res);
        }
    }

    return ret;
}

XCamReturn
ImageProcessor::push_3a_result (SmartPtr<X3aResult> &result)
{
    X3aResultList valid_results;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    if (!can_process_result(result))
        return XCAM_RETURN_BYPASS;

    ret = apply_3a_result (result);

    if (ret != XCAM_RETURN_NO_ERROR && ret != XCAM_RETURN_BYPASS) {
        XCAM_LOG_WARNING ("processor(%s) apply result failed", XCAM_STR(get_name()));
        return ret;
    }

    if (_callback) {
        _callback->process_image_result_done (this, result);
    }

    return ret;
}

void
ImageProcessor::filter_valid_results (X3aResultList &input, X3aResultList &valid_results)
{
    for (X3aResultList::iterator i_res = input.begin(); i_res != input.end(); ) {
        SmartPtr<X3aResult> &res = *i_res;
        if (can_process_result(res)) {
            valid_results.push_back (res);
            input.erase (i_res++);
        } else
            ++i_res;
    }
}

XCamReturn ImageProcessor::buffer_process_loop ()
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    SmartPtr<VideoBuffer> new_buf;
    SmartPtr<VideoBuffer> buf = _video_buf_queue.pop();

    if (!buf.ptr())
        return XCAM_RETURN_ERROR_MEM;

    ret = this->process_buffer (buf, new_buf);
    if (ret != XCAM_RETURN_NO_ERROR) {
        XCAM_LOG_DEBUG ("processing buffer failed");
    }

    if (_callback) {
        if (ret == XCAM_RETURN_NO_ERROR)
            _callback->process_buffer_done (this, new_buf);
        else
            _callback->process_buffer_failed (this, buf);
    }

    return XCAM_RETURN_NO_ERROR;
}


void
ImageProcessor::emit_stop ()
{
}

};
