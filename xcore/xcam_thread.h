/*
 * xcam_thread.h - Thread
 *
 *  Copyright (c) 2014 Intel Corporation
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

#ifndef XCAM_THREAD_H
#define XCAM_THREAD_H

#include <xcam_std.h>
#include <xcam_mutex.h>

namespace XCam {

class Thread {
public:
    Thread (const char *name = NULL);
    virtual ~Thread ();

    bool start ();
    virtual bool emit_stop ();
    bool stop ();
    bool is_running ();
    const char *get_name () const {
        return _name;
    }

protected:
    // return true to start loop, else the thread stopped
    virtual bool started ();
    virtual void stopped ();
    // return true to continue; false to stop
    virtual bool loop () = 0;
private:
    XCAM_DEAD_COPY (Thread);


private:
    static int thread_func (void *user_data);

private:
    char           *_name;
    pthread_t       _thread_id;
    XCam::Mutex     _mutex;
    XCam::Cond      _exit_cond;
    bool            _started;
    bool            _stopped;
};

};

#endif //XCAM_THREAD_H
