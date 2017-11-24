/*
 * cl_image_processor.h - CL image processor
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

#ifndef XCAM_CL_IMAGE_PROCESSOR_H
#define XCAM_CL_IMAGE_PROCESSOR_H

#include <xcam_std.h>
#include <image_processor.h>
#include <ocl/priority_buffer_queue.h>
#include <list>

namespace XCam {

class CLImageHandler;
class CLContext;
class CLHandlerThread;
class CLBufferNotifyThread;

class CLImageProcessor
    : public ImageProcessor
{
public:
    typedef std::list<SmartPtr<CLImageHandler>>  ImageHandlerList;
    typedef std::list<SmartPtr<PriorityBuffer>>  UnsafePriorityBufferList;
    friend class CLHandlerThread;
    friend class CLBufferNotifyThread;

public:
    explicit CLImageProcessor (const char* name = NULL);
    virtual ~CLImageProcessor ();

    void keep_attached_buf (bool flag);

    bool add_handler (SmartPtr<CLImageHandler> &handler);
    ImageHandlerList::iterator handlers_begin ();
    ImageHandlerList::iterator handlers_end ();

protected:

    //derive from ImageProcessor
    virtual bool can_process_result (SmartPtr<X3aResult> &result);
    virtual XCamReturn apply_3a_results (X3aResultList &results);
    virtual XCamReturn apply_3a_result (SmartPtr<X3aResult> &result);
    virtual XCamReturn process_buffer (SmartPtr<VideoBuffer> &input, SmartPtr<VideoBuffer> &output);
    virtual XCamReturn emit_start ();
    virtual void emit_stop ();

    SmartPtr<CLContext> get_cl_context ();

private:
    virtual XCamReturn create_handlers ();

    XCamReturn process_cl_buffer_queue ();
    XCamReturn process_done_buffer ();
    uint32_t check_ready_buffers ();

    XCAM_DEAD_COPY (CLImageProcessor);

protected:

// STREAM_LOCK only used in class derived from CLImageProcessor
#define STREAM_LOCK SmartLock stream_lock (this->_stream_mutex)
    // stream lock
    Mutex                          _stream_mutex;

private:
    SmartPtr<CLContext>            _context;
    ImageHandlerList               _handlers;
    SmartPtr<CLHandlerThread>      _handler_thread;
    PriorityBufferQueue            _process_buffer_queue;
    UnsafePriorityBufferList       _not_ready_buffers;
    SmartPtr<CLBufferNotifyThread> _done_buf_thread;
    SafeList<VideoBuffer>          _done_buffer_queue;
    uint32_t                       _seq_num;
    bool                           _keep_attached_buffer;  //default false
    XCAM_OBJ_PROFILING_DEFINES;
};

};
#endif //XCAM_CL_IMAGE_PROCESSOR_H
