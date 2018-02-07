/*
 * cl_image_processor.cpp - CL image processor
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
 */
#include "cl_image_processor.h"
#include "cl_context.h"
#include "cl_device.h"
#include "cl_image_handler.h"
#include "cl_demo_handler.h"
#include "xcam_thread.h"

namespace XCam {

class CLHandlerThread
    : public Thread
{
public:
    CLHandlerThread (CLImageProcessor *processor)
        : Thread ("CLHandlerThread")
        , _processor (processor)
    {}
    ~CLHandlerThread () {}

    virtual bool loop ();

private:
    CLImageProcessor *_processor;
};

bool CLHandlerThread::loop ()
{
    XCAM_ASSERT (_processor);
    XCamReturn ret = _processor->process_cl_buffer_queue ();
    if (ret != XCAM_RETURN_NO_ERROR && ret != XCAM_RETURN_BYPASS)
        return false;
    return true;
}

class CLBufferNotifyThread
    : public Thread
{
public:
    CLBufferNotifyThread (CLImageProcessor *processor)
        : Thread ("CLBufNtfThrd")
        , _processor (processor)
    {}
    ~CLBufferNotifyThread () {}

    virtual bool loop ();

private:
    CLImageProcessor *_processor;
};

bool CLBufferNotifyThread::loop ()
{
    XCAM_ASSERT (_processor);
    XCamReturn ret = _processor->process_done_buffer ();
    if (ret < XCAM_RETURN_NO_ERROR)
        return false;
    return true;
}
CLImageProcessor::CLImageProcessor (const char* name)
    : ImageProcessor (name ? name : "CLImageProcessor")
    , _seq_num (0)
    , _keep_attached_buffer (false)
{
    _context = CLDevice::instance ()->get_context ();
    XCAM_ASSERT (_context.ptr());

    SmartPtr<CLHandlerThread> handler_thread = new CLHandlerThread (this);
    XCAM_ASSERT (handler_thread.ptr ());
    _handler_thread = handler_thread;

    SmartPtr<CLBufferNotifyThread> done_buf_thread = new CLBufferNotifyThread (this);
    XCAM_ASSERT (done_buf_thread.ptr ());
    _done_buf_thread = done_buf_thread;

    XCAM_LOG_DEBUG ("CLImageProcessor constructed");
    XCAM_OBJ_PROFILING_INIT;
}

CLImageProcessor::~CLImageProcessor ()
{
    XCAM_LOG_DEBUG ("CLImageProcessor destructed");
}

void
CLImageProcessor::keep_attached_buf(bool flag)
{
    _keep_attached_buffer = flag;
}

bool
CLImageProcessor::add_handler (SmartPtr<CLImageHandler> &handler)
{
    XCAM_ASSERT (handler.ptr ());
    _handlers.push_back (handler);
    return true;
}

CLImageProcessor::ImageHandlerList::iterator
CLImageProcessor::handlers_begin ()
{
    return _handlers.begin ();
}

CLImageProcessor::ImageHandlerList::iterator
CLImageProcessor::handlers_end ()
{
    return _handlers.end ();
}

SmartPtr<CLContext>
CLImageProcessor::get_cl_context ()
{
    return _context;
}

bool
CLImageProcessor::can_process_result (SmartPtr<X3aResult> &result)
{
    XCAM_UNUSED (result);
    return false;
}

XCamReturn
CLImageProcessor::apply_3a_results (X3aResultList &results)
{
    XCAM_UNUSED (results);
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CLImageProcessor::apply_3a_result (SmartPtr<X3aResult> &result)
{
    XCAM_UNUSED (result);
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CLImageProcessor::process_buffer (SmartPtr<VideoBuffer> &input, SmartPtr<VideoBuffer> &output)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    XCAM_ASSERT (input.ptr ());

    // Always set to NULL,  output buf should be handled in CLBufferNotifyThread
    output = NULL;

    STREAM_LOCK;

    if (_handlers.empty()) {
        ret = create_handlers ();
    }

    XCAM_FAIL_RETURN (
        WARNING,
        !_handlers.empty () && ret == XCAM_RETURN_NO_ERROR,
        XCAM_RETURN_ERROR_CL,
        "CL image processor create handlers failed");

    SmartPtr<PriorityBuffer> p_buf = new PriorityBuffer;
    p_buf->set_seq_num (_seq_num++);
    p_buf->data = input;
    p_buf->handler = *(_handlers.begin ());

    XCAM_FAIL_RETURN (
        WARNING,
        _process_buffer_queue.push_priority_buf (p_buf),
        XCAM_RETURN_ERROR_UNKNOWN,
        "CLImageProcessor push priority buffer failed");

    return XCAM_RETURN_BYPASS;
}

XCamReturn
CLImageProcessor::process_done_buffer ()
{
    SmartPtr<VideoBuffer> done_buf = _done_buffer_queue.pop (-1);
    if (!done_buf.ptr ())
        return XCAM_RETURN_ERROR_THREAD;

    //notify buffer done, only in this thread
    notify_process_buffer_done (done_buf);
    return XCAM_RETURN_NO_ERROR;
}

uint32_t
CLImageProcessor::check_ready_buffers ()
{
    uint32_t ready_count = 0;
    bool is_ready_or_disabled = false;
    UnsafePriorityBufferList::iterator i = _not_ready_buffers.begin ();

    while (i != _not_ready_buffers.end()) {
        SmartPtr<PriorityBuffer> buf = *i;
        XCAM_ASSERT (buf.ptr () && buf->handler.ptr ());
        {
            is_ready_or_disabled = (!buf->handler->is_handler_enabled () || buf->handler->is_ready ());
        }
        if (is_ready_or_disabled) {
            ready_count ++;
            _process_buffer_queue.push_priority_buf (buf);
            _not_ready_buffers.erase (i++);
        } else
            ++i;
    }
    return ready_count;
}

XCamReturn
CLImageProcessor::process_cl_buffer_queue ()
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    SmartPtr<PriorityBuffer> p_buf;
    const int32_t timeout = 5000; // 5ms
    uint32_t ready_count = 0;

    {
        STREAM_LOCK;  // make sure handler APIs are protected
        check_ready_buffers ();
    }

    p_buf = _process_buffer_queue.pop (timeout);

    if (!p_buf.ptr ()) {
        //XCAM_LOG_DEBUG ("cl buffer queue stopped");
        return XCAM_RETURN_BYPASS;
    }

    SmartPtr<VideoBuffer> data = p_buf->data;
    SmartPtr<CLImageHandler> handler = p_buf->handler;
    SmartPtr <VideoBuffer> out_data;

    XCAM_ASSERT (data.ptr () && handler.ptr ());

    XCAM_LOG_DEBUG ("buf:%d, rank:%d\n", p_buf->seq_num, p_buf->rank);

    {
        STREAM_LOCK;
        if (handler->is_handler_enabled () && !handler->is_ready ()) {
            _not_ready_buffers.push_back (p_buf);
            return XCAM_RETURN_NO_ERROR;
        }

        ready_count = check_ready_buffers ();
        if (ready_count) {
            _process_buffer_queue.push_priority_buf (p_buf);
            return XCAM_RETURN_BYPASS;
        }

        ret = handler->execute (data, out_data);
        XCAM_FAIL_RETURN (
            WARNING,
            (ret == XCAM_RETURN_NO_ERROR || ret == XCAM_RETURN_BYPASS),
            ret,
            "CLImageProcessor execute image handler failed");
        XCAM_ASSERT (out_data.ptr ());
        if (ret == XCAM_RETURN_BYPASS)
            return ret;

        // for loop in handler, find next handler
        ImageHandlerList::iterator i_handler = _handlers.begin ();
        while (i_handler != _handlers.end ())
        {
            if (handler.ptr () == (*i_handler).ptr ()) {
                ++i_handler;
                break;
            }
            ++i_handler;
        }

        //skip all disabled handlers
        while (i_handler != _handlers.end () && !(*i_handler)->is_handler_enabled ())
            ++i_handler;

        if (i_handler != _handlers.end ())
            p_buf->handler = *i_handler;
        else
            p_buf->handler = NULL;
    }

    // buffer processed by all handlers, done
    if (!p_buf->handler.ptr ()) {
        if (!_keep_attached_buffer && out_data.ptr ())
            out_data->clear_attached_buffers ();

        XCAM_OBJ_PROFILING_START;
        CLDevice::instance()->get_context ()->finish ();
        XCAM_OBJ_PROFILING_END (get_name (), XCAM_OBJ_DUR_FRAME_NUM);

        // buffer done, push back
        _done_buffer_queue.push (out_data);
        return XCAM_RETURN_NO_ERROR;
    }

    p_buf->data = out_data;
    p_buf->down_rank ();

    XCAM_FAIL_RETURN (
        WARNING,
        _process_buffer_queue.push_priority_buf (p_buf),
        XCAM_RETURN_ERROR_UNKNOWN,
        "CLImageProcessor push priority buffer failed");

    return ret;
}

XCamReturn
CLImageProcessor::emit_start ()
{
    _done_buffer_queue.resume_pop ();
    _process_buffer_queue.resume_pop ();

    if (!_done_buf_thread->start ())
        return XCAM_RETURN_ERROR_THREAD;

    if (!_handler_thread->start ())
        return XCAM_RETURN_ERROR_THREAD;

    return XCAM_RETURN_NO_ERROR;
}

void
CLImageProcessor::emit_stop ()
{
    _process_buffer_queue.pause_pop();
    _done_buffer_queue.pause_pop ();


    for (ImageHandlerList::iterator i_handler = _handlers.begin ();
            i_handler != _handlers.end ();  ++i_handler) {
        (*i_handler)->emit_stop ();
    }

    _handler_thread->stop ();
    _done_buf_thread->stop ();
    _not_ready_buffers.clear ();
    _process_buffer_queue.clear ();
    _done_buffer_queue.clear ();
}

XCamReturn
CLImageProcessor::create_handlers ()
{
    SmartPtr<CLImageHandler> demo_handler;
    demo_handler = create_cl_demo_image_handler (_context);
    // demo_handler = create_cl_binary_demo_image_handler (_context);
    XCAM_FAIL_RETURN (
        WARNING,
        demo_handler.ptr (),
        XCAM_RETURN_ERROR_CL,
        "CLImageProcessor create demo handler failed");
    add_handler (demo_handler);

    return XCAM_RETURN_NO_ERROR;
}

};
