/*
 * x3a_image_process_center.h - 3a process center
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

#ifndef XCAM_3A_IMAGE_PROCESS_CENTER_H
#define XCAM_3A_IMAGE_PROCESS_CENTER_H

#include "xcam_utils.h"
#include "image_processor.h"

namespace XCam {

class X3aImageProcessCenter
    : public ImageProcessCallback
{
    typedef std::list<SmartPtr<ImageProcessor> > ImageProcessorList;
    typedef std::list<SmartPtr<ImageProcessor> >::iterator ImageProcessorIter;
public:
    explicit X3aImageProcessCenter();
    ~X3aImageProcessCenter();

    bool insert_processor (SmartPtr<ImageProcessor> &processor);
    bool has_processors ();
    bool set_image_callback (ImageProcessCallback *callback);

    XCamReturn start ();
    XCamReturn stop ();

    bool put_buffer (SmartPtr<VideoBuffer> &buf);

    XCamReturn put_3a_results (X3aResultList &results);
    XCamReturn put_3a_result (SmartPtr<X3aResult> &result);

    //derived from ImageProcessCallback
    virtual void process_buffer_done (ImageProcessor *processor, const SmartPtr<VideoBuffer> &buf);
    virtual void process_buffer_failed (ImageProcessor *processor, const SmartPtr<VideoBuffer> &buf);
    virtual void process_image_result_done (ImageProcessor *processor, const SmartPtr<X3aResult> &result);

private:
    XCAM_DEAD_COPY (X3aImageProcessCenter);

private:
    ImageProcessorList             _image_processors;
    ImageProcessCallback          *_callback;
};

};
#endif //XCAM_3A_IMAGE_PROCESS_CENTER_H
