/*
 * worker.h - worker class interface
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

#ifndef XCAM_WORKER_H
#define XCAM_WORKER_H

#include <xcam_std.h>

#define ENABLE_FUNC_OBJ 0

#define DECLARE_WORK_CALLBACK(CbClass, Handler, mem_func)                 \
    class CbClass : public ::XCam::Worker::Callback {                     \
        private: ::XCam::SmartPtr<Handler>  _h;                           \
        public: CbClass (const ::XCam::SmartPtr<Handler> &h) { _h = h;}   \
        protected: void work_status (                                     \
            const ::XCam::SmartPtr<::XCam::Worker> &worker,               \
            const ::XCam::SmartPtr<::XCam::Worker::Arguments> &args,      \
            const XCamReturn error) {                                     \
            _h->mem_func (worker, args, error);  }                        \
    }

namespace XCam {

class Worker
    : public RefObj
{
public:
    struct Arguments
    {
        Arguments () {}
        virtual ~Arguments () {}

        XCAM_DEAD_COPY (Arguments);
    };

    class Callback {
    public:
        Callback () {}
        virtual ~Callback () {}

        virtual void work_status (
            const SmartPtr<Worker> &worker, const SmartPtr<Arguments> &args, const XCamReturn error) = 0;

    private:
        XCAM_DEAD_COPY (Callback);
    };

#if ENABLE_FUNC_OBJ
    class FuncObj {
    public:
        virtual ~FuncObj () {}
        virtual XCamReturn impl (const SmartPtr<Arguments> &args) = 0;

    private:
        XCAM_DEAD_COPY (FuncObj);
    };
#endif

protected:
    explicit Worker (const char *name, const SmartPtr<Callback> &cb = NULL);

public:
    virtual ~Worker ();
    bool set_name (const char *name);
    const char *get_name () const {
        return _name;
    }
#if ENABLE_FUNC_OBJ
    bool set_func_obj (const SmartPtr<FuncObj> &obj);
#endif
    bool set_callback (const SmartPtr<Callback> &callback);

    virtual XCamReturn work (const SmartPtr<Arguments> &args) = 0;
    virtual XCamReturn stop () = 0;

protected:
    virtual void status_check (const SmartPtr<Arguments> &args, const XCamReturn error);

private:
    XCAM_DEAD_COPY (Worker);

private:
    char                      *_name;
    SmartPtr<Callback>         _callback;
#if ENABLE_FUNC_OBJ
    SmartPtr<FuncObj>          _func_obj;
#endif
};

}
#endif //XCAM_WORKER_H
