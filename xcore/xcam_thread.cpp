/*
 * xcam_thread.cpp - Thread
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

#include "xcam_thread.h"
#include "xcam_mutex.h"
#include <errno.h>

namespace XCam {

Thread::Thread (const char *name)
    : _name (NULL)
    , _thread_id (0)
    , _started (false)
    , _stopped (true)
{
    if (name)
        _name = strndup (name, XCAM_MAX_STR_SIZE);
}

Thread::~Thread ()
{
    if (_name)
        xcam_free (_name);
}

int
Thread::thread_func (void *user_data)
{
    Thread *thread = (Thread *)user_data;
    bool ret = true;

    {
        // Make sure running after start
        SmartLock locker(thread->_mutex);
        pthread_detach (pthread_self());
    }
    ret = thread->started ();

    while (true) {
        {
            SmartLock locker(thread->_mutex);
            if (!thread->_started || ret == false) {
                thread->_started = false;
                thread->_thread_id = 0;
                ret = false;
                break;
            }
        }

        ret = thread->loop ();
    }

    thread->stopped ();

    SmartLock locker(thread->_mutex);
    thread->_stopped = true;
    thread->_exit_cond.broadcast ();

    return 0;
}

bool
Thread::started ()
{
    XCAM_LOG_DEBUG ("Thread(%s) started", XCAM_STR(_name));
    return true;
}

void
Thread::stopped ()
{
    XCAM_LOG_DEBUG ("Thread(%s) stopped", XCAM_STR(_name));
}

bool Thread::start ()
{
    SmartLock locker(_mutex);
    if (_started)
        return true;

    if (pthread_create (&_thread_id, NULL, (void * (*)(void*))thread_func, this) != 0)
        return false;
    _started = true;
    _stopped = false;

#ifdef __USE_GNU
    char thread_name[16];
    xcam_mem_clear (thread_name);
    snprintf (thread_name, sizeof (thread_name), "xc:%s", XCAM_STR(_name));
    int ret = pthread_setname_np (_thread_id, thread_name);
    if (ret != 0) {
        XCAM_LOG_WARNING ("Thread(%s) set name to thread_id failed.(%d, %s)", XCAM_STR(_name), ret, strerror(ret));
    }
#endif

    return true;
}

bool
Thread::emit_stop ()
{
    SmartLock locker(_mutex);
    _started = false;
    return true;
}

bool Thread::stop ()
{
    SmartLock locker(_mutex);
    if (_started) {
        _started = false;
    }
    if (!_stopped) {
        _exit_cond.wait(_mutex);
    }
    return true;
}

bool Thread::is_running ()
{
    SmartLock locker(_mutex);
    return _started;
}

};
