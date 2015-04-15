/*
 * xcam_SmartPtr.h - start pointer
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
#ifndef XCAM_SMARTPTR_H
#define XCAM_SMARTPTR_H

#include <stdint.h>
#include <atomic>
#include <base/xcam_defs.h>

namespace XCam {

class RefCount {
public:
    RefCount (): _ref_count(1) {}
    void ref() {
        ++_ref_count;
    }
    uint32_t unref() {
        return --_ref_count;
    }
private:
    mutable std::atomic<uint32_t> _ref_count;
};


template <typename Obj>
class SmartPtr {
private:
    template<typename ObjDerive> friend class SmartPtr;
public:
    SmartPtr (Obj *obj = NULL) : _ptr (obj), _ref(NULL) {
        if (_ptr)
            _ref = new RefCount();
    }
    template <typename ObjDerive>
    SmartPtr (ObjDerive *obj) : _ptr (obj), _ref(NULL) {
        if (_ptr)
            _ref = new RefCount();
    }

    // copy from pointer
    SmartPtr (const SmartPtr<Obj> &obj)
        : _ptr(obj._ptr), _ref(obj._ref)  {
        if (_ptr)
            _ref->ref();
    }
    template <typename ObjDerive>
    SmartPtr (const SmartPtr<ObjDerive> &obj)
        : _ptr(obj._ptr), _ref(obj._ref)  {
        if (_ptr)
            _ref->ref();
    }
    ~SmartPtr () {
        release();
    }

    /* operator = */
    SmartPtr<Obj> & operator = (Obj *obj) {
        release ();
        new_pointer (obj, NULL);
        return *this;
    }
    template <typename ObjDerive>
    SmartPtr<Obj> & operator = (ObjDerive *obj) {
        release ();
        new_pointer (obj, NULL);
        return *this;
    }
    SmartPtr<Obj> & operator = (const SmartPtr<Obj> &obj) {
        release ();
        new_pointer (obj._ptr, obj._ref);
        return *this;
    }
    template <typename ObjDerive>
    SmartPtr<Obj> & operator = (const SmartPtr<ObjDerive> &obj) {
        release ();
        new_pointer (obj._ptr, obj._ref);
        return *this;
    }

    Obj *operator -> () const {
        return _ptr;
    }

    Obj *ptr() const {
        return _ptr;
    }

    void release() {
        if (!_ptr)
            return;
        XCAM_ASSERT (_ref);
        if (!_ref->unref()) {
            delete _ref;
            delete _ptr;
        }
        _ptr = NULL;
        _ref = NULL;
    }

    template <typename ObjDerive>
    SmartPtr<ObjDerive> dynamic_cast_ptr () const {
        SmartPtr<ObjDerive> ret(NULL);
        ObjDerive *obj_derive(NULL);
        if (!_ref)
            return ret;
        obj_derive = dynamic_cast<ObjDerive*>(_ptr);
        if (!obj_derive)
            return ret;
        ret.new_pointer (obj_derive, _ref);
        return ret;
    }
private:
    void new_pointer (Obj *obj, RefCount *ref) {
        if (!obj) {
            _ptr = NULL;
            _ref = NULL;
        }
        _ptr = obj;
        if (ref) {
            _ref = ref;
            _ref->ref();
        } else
            _ref = new RefCount();
    }

private:

    Obj      *_ptr;
    mutable RefCount *_ref;
};

}; // end namespace
#endif //XCAM_SMARTPTR_H