/*
 * cl_multi_image_handler.h - CL multi-image handler
 *
 *  Copyright (c) 2016 Intel Corporation
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

#ifndef XCAM_CL_MULTI_IMAGE_HANDLER_H
#define XCAM_CL_MULTI_IMAGE_HANDLER_H

#include <xcam_std.h>
#include <ocl/cl_image_handler.h>

namespace XCam {

class CLMultiImageHandler
    : public CLImageHandler
{
public:
    typedef std::list<SmartPtr<CLImageHandler> > HandlerList;

public:
    explicit CLMultiImageHandler (const SmartPtr<CLContext> &context, const char *name);
    virtual ~CLMultiImageHandler ();
    bool add_image_handler (SmartPtr<CLImageHandler> &handler);

protected:
    virtual XCamReturn prepare_parameters (SmartPtr<VideoBuffer> &input, SmartPtr<VideoBuffer> &output);
    virtual XCamReturn execute_kernels ();
    virtual XCamReturn execute_done (SmartPtr<VideoBuffer> &output);

    virtual XCamReturn sub_handler_execute_done (SmartPtr<CLImageHandler> &handler);

    XCamReturn ensure_handler_parameters (
        const SmartPtr<CLImageHandler> &handler, SmartPtr<VideoBuffer> &input, SmartPtr<VideoBuffer> &output);

private:
    XCAM_DEAD_COPY (CLMultiImageHandler);

protected:
    HandlerList        _handler_list;
};

};

#endif // XCAM_CL_MULTI_IMAGE_HANDLER_H