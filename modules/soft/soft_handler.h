/*
 * soft_handler.h - soft image handler class
 *
 *  Copyright (c) 2017 Intel Corporation
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

#ifndef XCAM_SOFT_HANDLER_H
#define XCAM_SOFT_HANDLER_H

#include "xcam_utils.h"
#include "image_handler.h"

namespace XCam {

class SoftHandler;
class ThreadPool;
class SyncMeta;

class SoftHandler
    : public ImageHandler
{
public:
    explicit SoftHandler (const char* name);
    ~SoftHandler ();

    bool set_threads (uint32_t num);

    // derive from ImageHandler
    virtual XCamReturn execute_buffer (SmartPtr<Parameters> &params, bool sync);
    virtual XCamReturn finish ();
    virtual XCamReturn terminate ();

protected:
    virtual XCamReturn configure_resource (const SmartPtr<Parameters> &params);
    virtual SmartPtr<Worker::Arguments> get_first_worker_args (SmartPtr<Parameters> &params) = 0;
    virtual XCamReturn last_worker_done (SmartPtr<Parameters> &params, XCamReturn err);

private:
    XCAM_DEAD_COPY (SoftHandler);

private:
    SmartPtr<ThreadPool>    _threads;
    SmartPtr<SyncMeta>      _cur_sync;
    bool                    _need_configure;
    mutable std::atomic<int32_t>  _wip_buf_count;
};

}

#endif //XCAM_SOFT_HANDLER_H
