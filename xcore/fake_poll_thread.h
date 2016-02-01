/*
 * fake_poll_thread.h - poll thread for raw image
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
 * Author: Jia Meng <jia.meng@intel.com>
 */

#ifndef XCAM_FAKE_POLL_THREAD_H
#define XCAM_FAKE_POLL_THREAD_H

#include "poll_thread.h"

namespace XCam {

class DrmBoBufferPool;
class DrmBoBuffer;

class FakePollThread
    : public PollThread
{
public:
    explicit FakePollThread (const char *raw_path);
    ~FakePollThread ();

    virtual XCamReturn start();
    virtual XCamReturn stop ();

protected:
    virtual XCamReturn poll_buffer_loop ();

private:
    XCAM_DEAD_COPY (FakePollThread);

    virtual XCamReturn init_3a_stats_pool () {
        return XCAM_RETURN_ERROR_UNKNOWN;
    }
    XCamReturn init_buffer_pool ();
    XCamReturn read_buf (SmartPtr<DrmBoBuffer> &buf);

private:
    char                        *_raw_path;
    FILE                        *_raw;
#if HAVE_LIBDRM
    SmartPtr<DrmBoBufferPool>    _buf_pool;
#endif
};

};

#endif //XCAM_FAKE_POLL_THREAD_H
