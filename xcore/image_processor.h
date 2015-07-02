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

#ifndef XCAM_IMAGE_PROCESSOR_H
#define XCAM_IMAGE_PROCESSOR_H

#include "xcam_utils.h"
#include "video_buffer.h"
#include "x3a_result.h"
#include "smartptr.h"
#include "safe_list.h"

namespace XCam {

class ImageProcessor;

/* callback interface */
class ImageProcessCallback {
public:
    ImageProcessCallback () {}
    virtual ~ImageProcessCallback () {}
    virtual void process_buffer_done (ImageProcessor *processor, const SmartPtr<VideoBuffer> &buf);
    virtual void process_buffer_failed (ImageProcessor *processor, const SmartPtr<VideoBuffer> &buf);
    virtual void process_image_result_done (ImageProcessor *processor, const SmartPtr<X3aResult> &result);

private:
    XCAM_DEAD_COPY (ImageProcessCallback);
};

class ImageProcessorThread;
class X3aResultsProcessThread;

/* base class, ImageProcessor */
class ImageProcessor
{
    friend class ImageProcessorThread;
    friend class X3aResultsProcessThread;

    typedef SafeList<VideoBuffer> VideoBufQueue;

public:
    explicit ImageProcessor (const char* name);
    virtual ~ImageProcessor ();

    const char *get_name () const {
        return _name;
    }

    bool set_callback (ImageProcessCallback *callback);
    XCamReturn start();
    XCamReturn stop ();

    XCamReturn push_buffer (SmartPtr<VideoBuffer> &buf);
    XCamReturn push_3a_results (X3aResultList &results);
    XCamReturn push_3a_result (SmartPtr<X3aResult> &result);

protected:
    virtual bool can_process_result (SmartPtr<X3aResult> &result) = 0;
    virtual XCamReturn apply_3a_results (X3aResultList &results) = 0;
    virtual XCamReturn apply_3a_result (SmartPtr<X3aResult> &result) = 0;
    // buffer runs in another thread
    virtual XCamReturn process_buffer(SmartPtr<VideoBuffer> &input, SmartPtr<VideoBuffer> &output) = 0;
    virtual XCamReturn emit_start ();
    virtual void emit_stop ();


    void notify_process_buffer_done (const SmartPtr<VideoBuffer> &buf);
    void notify_process_buffer_failed (const SmartPtr<VideoBuffer> &buf);

private:
    void filter_valid_results (X3aResultList &input, X3aResultList &valid_results);
    XCamReturn buffer_process_loop ();

    XCamReturn process_3a_results (X3aResultList &results);
    XCamReturn process_3a_result (SmartPtr<X3aResult> &result);

private:
    XCAM_DEAD_COPY (ImageProcessor);

protected:
    char                               *_name;
    ImageProcessCallback               *_callback;
    SmartPtr<ImageProcessorThread>      _processor_thread;
    VideoBufQueue                       _video_buf_queue;
    SmartPtr<X3aResultsProcessThread>   _results_thread;
};

};

#endif //XCAM_IMAGE_PROCESSOR_H
