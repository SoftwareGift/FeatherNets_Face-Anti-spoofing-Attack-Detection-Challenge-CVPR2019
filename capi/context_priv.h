/*
 * context_priv.h - capi private context
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

#ifndef XCAM_CONTEXT_PRIV_H
#define XCAM_CONTEXT_PRIV_H

#include <xcam_utils.h>
#include <string.h>
#include <ocl/cl_image_handler.h>
#include <ocl/cl_context.h>
#include <ocl/cl_blender.h>
#include <interface/stitcher.h>

using namespace XCam;

enum HandleType {
    HandleTypeNone = 0,
    HandleType3DNR,
    HandleTypeWaveletNR,
    HandleTypeFisheye,
    HandleTypeDefog,
    HandleTypeDVS,
    HandleTypeStitch,
};

#define CONTEXT_CAST(Type, handle) (Type*)(handle)
#define CONTEXT_BASE_CAST(handle) (ContextBase*)(handle)
#define HANDLE_CAST(context) (XCamHandle*)(context)

bool handle_name_equal (const char *name, HandleType type);

typedef struct _CompareStr {
    bool operator() (const char* str1, const char* str2) const {
        return strncmp(str1, str2, 1024) < 0;
    }
} CompareStr;

typedef std::map<const char*, const char*, CompareStr> ContextParams;

class ContextBase {
public:
    virtual ~ContextBase ();

    virtual XCamReturn set_parameters (ContextParams &param_list);
    virtual const char* get_usage () const {
        return _usage;
    }
    XCamReturn init_handler ();
    XCamReturn uinit_handler ();

    XCamReturn execute (SmartPtr<DrmBoBuffer> &buf_in, SmartPtr<DrmBoBuffer> &buf_out);

    SmartPtr<CLImageHandler> get_handler() const {
        return  _handler;
    }
    SmartPtr<DrmBoBufferPool> get_input_buffer_pool() const {
        return  _inbuf_pool;
    }
    HandleType get_type () const {
        return _type;
    }
    const char* get_type_name () const;

protected:
    ContextBase (HandleType type);
    void set_handler (const SmartPtr<CLImageHandler> &ptr) {
        _handler = ptr;
    }

    virtual SmartPtr<CLImageHandler> create_handler (SmartPtr<CLContext> &context) = 0;

private:
    XCAM_DEAD_COPY (ContextBase);

protected:
    HandleType                       _type;
    char                            *_usage;
    SmartPtr<CLImageHandler>         _handler;
    SmartPtr<DrmBoBufferPool>        _inbuf_pool;

    //parameters
    uint32_t                         _image_width;
    uint32_t                         _image_height;
    bool                             _alloc_out_buf;
};

class NR3DContext
    : public ContextBase
{
public:
    NR3DContext ()
        : ContextBase (HandleType3DNR)
    {}

    virtual SmartPtr<CLImageHandler> create_handler (SmartPtr<CLContext> &context);
};

class NRWaveletContext
    : public ContextBase
{
public:
    NRWaveletContext ()
        : ContextBase (HandleTypeWaveletNR)
    {}

    virtual SmartPtr<CLImageHandler> create_handler (SmartPtr<CLContext> &context);
};

class FisheyeContext
    : public ContextBase
{
public:
    FisheyeContext ()
        : ContextBase (HandleTypeFisheye)
    {}

    virtual SmartPtr<CLImageHandler> create_handler (SmartPtr<CLContext> &context);
};

class DefogContext
    : public ContextBase
{
public:
    DefogContext ()
        : ContextBase (HandleTypeDefog)
    {}

    virtual SmartPtr<CLImageHandler> create_handler (SmartPtr<CLContext> &context);
};

class DVSContext
    : public ContextBase
{
public:
    DVSContext ()
        : ContextBase (HandleTypeDVS)
    {}

    virtual SmartPtr<CLImageHandler> create_handler (SmartPtr<CLContext> &context);
};

class StitchContext
    : public ContextBase
{
public:
    StitchContext ()
        : ContextBase (HandleTypeStitch)
        , _need_seam (false)
        , _fisheye_map (false)
        , _need_lsc (false)
        , _fm_ocl (false)
        , _scale_mode (CLBlenderScaleLocal)
        , _res_mode (StitchRes1080P)
    {}

    virtual SmartPtr<CLImageHandler> create_handler (SmartPtr<CLContext> &context);

private:
    bool                  _need_seam;
    bool                  _fisheye_map;
    bool                  _need_lsc;
    bool                  _fm_ocl;
    CLBlenderScaleMode    _scale_mode;
    StitchResMode         _res_mode;
};

#endif //XCAM_CONTEXT_PRIV_H
