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

#include <xcam_std.h>
#include <image_handler.h>
#include <video_buffer.h>
#include <worker.h>

namespace XCam {

class SoftHandler;
class ThreadPool;
class SyncMeta;
class SoftWorker;

struct SoftArgs
    : Worker::Arguments
{
private:
    SmartPtr<ImageHandler::Parameters> _param;
public:
    explicit SoftArgs (const SmartPtr<ImageHandler::Parameters> &param = NULL) : _param (param) {}
    inline const SmartPtr<ImageHandler::Parameters> &get_param () const {
        return _param;
    }
    inline void set_param (const SmartPtr<ImageHandler::Parameters> &param) {
        _param = param;
        XCAM_ASSERT (param.ptr ());
    }
};

class SoftHandler
    : public ImageHandler
{
public:
    explicit SoftHandler (const char* name);
    ~SoftHandler ();

    bool set_threads (const SmartPtr<ThreadPool> &pool);
    bool set_out_video_info (const VideoBufferInfo &info);
    bool enable_allocator (bool enable);

    // derive from ImageHandler
    virtual XCamReturn execute_buffer (const SmartPtr<Parameters> &param, bool sync);
    virtual XCamReturn finish ();
    virtual XCamReturn terminate ();

protected:
    virtual XCamReturn configure_resource (const SmartPtr<Parameters> &param) = 0;
    virtual XCamReturn start_work (const SmartPtr<Parameters> &param) = 0;
    //virtual SmartPtr<Worker::Arguments> get_first_worker_args (const SmartPtr<SoftWorker> &worker, SmartPtr<Parameters> &params) = 0;
    virtual void work_well_done (const SmartPtr<ImageHandler::Parameters> &param, XCamReturn err);
    virtual void work_broken (const SmartPtr<ImageHandler::Parameters> &param, XCamReturn err);

    //directly usage
    bool check_work_continue (const SmartPtr<ImageHandler::Parameters> &param, XCamReturn err);

private:
    XCamReturn confirm_configured ();
    void param_ended (SmartPtr<ImageHandler::Parameters> param, XCamReturn err);
    static bool is_param_error (const SmartPtr<ImageHandler::Parameters> &param);

private:
    XCAM_DEAD_COPY (SoftHandler);

private:
    SmartPtr<ThreadPool>    _threads;
    VideoBufferInfo         _out_video_info;
    SmartPtr<SyncMeta>      _cur_sync;
    bool                    _need_configure;
    bool                    _enable_allocator;
    SafeList<Parameters>    _params;
    mutable std::atomic<int32_t>  _wip_buf_count;
};

}

#endif //XCAM_SOFT_HANDLER_H
