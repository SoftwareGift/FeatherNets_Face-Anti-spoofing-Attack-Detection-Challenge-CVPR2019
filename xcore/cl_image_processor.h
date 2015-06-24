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

#include "xcam_utils.h"
#include "image_processor.h"
#include <list>

namespace XCam {

class CLImageHandler;
class CLContext;

class CLImageProcessor
    : public ImageProcessor
{
    typedef std::list<SmartPtr<CLImageHandler>>  ImageHandlerList;
public:
    explicit CLImageProcessor (const char* name = NULL);
    virtual ~CLImageProcessor ();

    bool add_handler (SmartPtr<CLImageHandler> &handler);

protected:

    //derive from ImageProcessor
    virtual bool can_process_result (SmartPtr<X3aResult> &result);
    virtual XCamReturn apply_3a_results (X3aResultList &results);
    virtual XCamReturn apply_3a_result (SmartPtr<X3aResult> &result);
    virtual XCamReturn process_buffer (SmartPtr<VideoBuffer> &input, SmartPtr<VideoBuffer> &output);
    virtual void emit_stop ();

    SmartPtr<CLContext> get_cl_context ();

private:
    virtual XCamReturn create_handlers ();
    XCAM_DEAD_COPY (CLImageProcessor);

protected:

// STREAM_LOCK only used in class derived from CLImageProcessor
#define STREAM_LOCK SmartLock stream_lock (this->_stream_mutex)
    // stream lock
    Mutex                          _stream_mutex;

private:
    SmartPtr<CLContext>            _context;
    ImageHandlerList               _handlers;
    XCAM_OBJ_PROFILING_DEFINES;
};

};
#endif //XCAM_CL_IMAGE_PROCESSOR_H
