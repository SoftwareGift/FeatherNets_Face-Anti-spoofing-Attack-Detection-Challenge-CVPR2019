/*
 * thread_pool.h - Thread Pool
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

#ifndef XCAM_THREAD_POOL_H
#define XCAM_THREAD_POOL_H

#include <xcam_std.h>
#include <safe_list.h>
#include <xcam_thread.h>

namespace XCam {

class UserThread;

class ThreadPool
    : public RefObj
{
    friend class UserThread;
    typedef std::list<SmartPtr<UserThread> > UserThreadList;

public:
    class UserData {
    public:
        UserData () {}
        virtual ~UserData () {}
        virtual XCamReturn run () = 0;
        virtual void done (XCamReturn) {}
    private:
        XCAM_DEAD_COPY (UserData);
    };

public:
    explicit ThreadPool (const char *name);
    virtual ~ThreadPool ();
    bool set_threads (uint32_t min, uint32_t max);
    const char *get_name () const {
        return _name;
    }
    bool is_running ();

    XCamReturn start ();
    XCamReturn stop ();
    XCamReturn queue (const SmartPtr<UserData> &data);

protected:
    bool dispatch (const SmartPtr<UserData> &data);
    XCamReturn create_user_thread_unsafe ();

private:
    XCAM_DEAD_COPY (ThreadPool);

private:
    char                   *_name;
    uint32_t                _min_threads;
    uint32_t                _max_threads;
    uint32_t                _allocated_threads;
    uint32_t                _free_threads;
    bool                    _running;
    UserThreadList          _thread_list;
    Mutex                   _mutex;

    SafeList<UserData>      _data_queue;
};

}

#endif // XCAM_THREAD_POOL_H
