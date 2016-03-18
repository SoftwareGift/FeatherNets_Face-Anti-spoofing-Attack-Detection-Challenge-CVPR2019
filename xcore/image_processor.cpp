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
ImageProcessCallback::process_buffer_done (ImageProcessor *processor, const SmartPtr<VideoBuffer> &buf) {
    XCAM_ASSERT (buf.ptr() && processor);

    int64_t ts = buf->get_timestamp();
    XCAM_UNUSED (ts);
    XCAM_LOG_DEBUG (
        "processor(%s) handled buffer(" XCAM_TIMESTAMP_FORMAT ") successfully",
        XCAM_STR(processor->get_name()),
        XCAM_TIMESTAMP_ARGS (ts));
}

void
ImageProcessCallback::process_buffer_failed (ImageProcessor *processor, const SmartPtr<VideoBuffer> &buf)
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
ImageProcessCallback::process_image_result_done (ImageProcessor *processor, const SmartPtr<X3aResult> &result)
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

class X3aResultsProcessThread
    : public Thread
{
    typedef SafeList<X3aResult> ResultQueue;
public:
    X3aResultsProcessThread (ImageProcessor *processor)
        : Thread ("x3a_results_process_thread")
        , _processor (processor)
    {}
    ~X3aResultsProcessThread () {}

    XCamReturn push_result (SmartPtr<X3aResult> &result) {
        _queue.push (result);
        return XCAM_RETURN_NO_ERROR;
    }

    void triger_stop () {
        _queue.pause_pop ();
    }

    virtual bool loop ();

private:
    ImageProcessor  *_processor;
    ResultQueue      _queue;
};

bool X3aResultsProcessThread::loop ()
{
    X3aResultList result_list;
    SmartPtr<X3aResult> result;

    result = _queue.pop (-1);
    if (!result.ptr ())
        return false;

    result_list.push_back (result);
    while ((result = _queue.pop (0)).ptr ()) {
        result_list.push_back (result);
    }

    XCamReturn ret = _processor->process_3a_results (result_list);
    if (ret != XCAM_RETURN_NO_ERROR) {
        XCAM_LOG_DEBUG ("processing 3a result failed");
    }

    return true;
}
ImageProcessor::ImageProcessor (const char* name)
    : _name (NULL)
    , _callback (NULL)
{
    if (name)
        _name = strndup (name, XCAM_MAX_STR_SIZE);

    _processor_thread = new ImageProcessorThread (this);
    _results_thread = new X3aResultsProcessThread (this);
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
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    if (!_results_thread->start ()) {
        return XCAM_RETURN_ERROR_THREAD;
    }
    if (!_processor_thread->start ()) {
        return XCAM_RETURN_ERROR_THREAD;
    }
    ret = emit_start ();
    if (ret != XCAM_RETURN_NO_ERROR) {
        XCAM_LOG_WARNING ("ImageProcessor(%s) emit start failed", XCAM_STR (_name));
        _video_buf_queue.pause_pop ();
        _results_thread->triger_stop ();
        _processor_thread->stop ();
        _results_thread->stop ();
        return ret;
    }
    XCAM_LOG_INFO ("ImageProcessor(%s) started", XCAM_STR (_name));
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
ImageProcessor::stop()
{
    _video_buf_queue.pause_pop ();
    _results_thread->triger_stop ();

    emit_stop ();

    _processor_thread->stop ();
    _results_thread->stop ();
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
    XCAM_ASSERT (!results.empty ());
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    for (X3aResultList::iterator i_res = results.begin();
            i_res != results.end(); ++i_res) {
        SmartPtr<X3aResult> &res = *i_res;

        ret = _results_thread->push_result (res);
        if (ret != XCAM_RETURN_NO_ERROR)
            break;
    }

    XCAM_FAIL_RETURN(
        WARNING,
        ret == XCAM_RETURN_NO_ERROR,
        ret,
        "processor(%s) push 3a results failed", XCAM_STR(get_name()));
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
ImageProcessor::push_3a_result (SmartPtr<X3aResult> &result)
{
    XCamReturn ret = _results_thread->push_result (result);
    XCAM_FAIL_RETURN(
        WARNING,
        ret == XCAM_RETURN_NO_ERROR,
        ret,
        "processor(%s) push 3a result failed", XCAM_STR(get_name()));
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
ImageProcessor::process_3a_results (X3aResultList &results)
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
ImageProcessor::process_3a_result (SmartPtr<X3aResult> &result)
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

void
ImageProcessor::notify_process_buffer_done (const SmartPtr<VideoBuffer> &buf)
{
    if (_callback)
        _callback->process_buffer_done (this, buf);
}

void
ImageProcessor::notify_process_buffer_failed (const SmartPtr<VideoBuffer> &buf)
{
    if (_callback)
        _callback->process_buffer_failed (this, buf);
}

XCamReturn
ImageProcessor::buffer_process_loop ()
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    SmartPtr<VideoBuffer> new_buf;
    SmartPtr<VideoBuffer> buf = _video_buf_queue.pop();

    if (!buf.ptr())
        return XCAM_RETURN_ERROR_MEM;

    ret = this->process_buffer (buf, new_buf);
    if (ret < XCAM_RETURN_NO_ERROR) {
        XCAM_LOG_DEBUG ("processing buffer failed");
        notify_process_buffer_failed (buf);
        return ret;
    }

    if (new_buf.ptr ())
        notify_process_buffer_done (new_buf);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
ImageProcessor::emit_start ()
{
    return XCAM_RETURN_NO_ERROR;
}

void
ImageProcessor::emit_stop ()
{
}

};
