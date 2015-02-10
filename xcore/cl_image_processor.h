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
    explicit CLImageProcessor ();
    virtual ~CLImageProcessor ();

protected:
    //derive from ImageProcessor
    virtual bool can_process_result (SmartPtr<X3aResult> &result);
    virtual XCamReturn apply_3a_results (X3aResultList &results);
    virtual XCamReturn apply_3a_result (SmartPtr<X3aResult> &result);
    virtual XCamReturn process_buffer (SmartPtr<VideoBuffer> &input, SmartPtr<VideoBuffer> &output);

private:
    virtual XCamReturn create_handlers ();
    XCAM_DEAD_COPY (CLImageProcessor);

private:
    SmartPtr<CLContext>            _context;
    ImageHandlerList               _handlers;
};

};
#endif //XCAM_CL_IMAGE_PROCESSOR_H
