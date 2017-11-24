/*
 * soft_blender.h - soft blender class
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

#ifndef XCAM_SOFT_BLENDER_H
#define XCAM_SOFT_BLENDER_H

#include <xcam_std.h>
#include <interface/blender.h>
#include <soft/soft_handler.h>

#define XCAM_SOFT_PYRAMID_MAX_LEVEL 4
#define XCAM_SOFT_PYRAMID_DEFAULT_LEVEL 3

namespace XCam {

namespace SoftBlenderPriv {
class BlenderPrivConfig;
};

class SoftBlender
    : public SoftHandler, public Blender
{
    friend class SoftBlenderPriv::BlenderPrivConfig;
    friend SmartPtr<SoftHandler> create_soft_blender ();
public:
    struct BlenderParam : ImageHandler::Parameters {
        SmartPtr<VideoBuffer> in1_buf;

        BlenderParam (
            const SmartPtr<VideoBuffer> &in0,
            const SmartPtr<VideoBuffer> &in1,
            const SmartPtr<VideoBuffer> &out)
            : Parameters (in0, out)
            , in1_buf (in1)
        {}
    };

    enum BufIdx {
        Idx0 = 0,
        Idx1,
        BufIdxCount,
    };

public:
    ~SoftBlender ();

    bool set_pyr_levels (uint32_t num);

    //derived from SoftHandler
    virtual XCamReturn terminate ();

    void gauss_scale_done (
        const SmartPtr<Worker> &worker, const SmartPtr<Worker::Arguments> &args, const XCamReturn error);
    void lap_done (
        const SmartPtr<Worker> &worker, const SmartPtr<Worker::Arguments> &args, const XCamReturn error);
    void blend_task_done (
        const SmartPtr<Worker> &worker, const SmartPtr<Worker::Arguments> &args, const XCamReturn error);
    void reconstruct_done (
        const SmartPtr<Worker> &worker, const SmartPtr<Worker::Arguments> &args, const XCamReturn error);

protected:
    explicit SoftBlender (const char *name = "SoftBlender");

    //derived from Blender interface
    XCamReturn blend (
        const SmartPtr<VideoBuffer> &in0,
        const SmartPtr<VideoBuffer> &in1,
        SmartPtr<VideoBuffer> &out_buf);

    //derived from SoftHandler
    XCamReturn configure_resource (const SmartPtr<Parameters> &param);
    XCamReturn start_work (const SmartPtr<Parameters> &param);

private:
    SmartPtr<SoftBlenderPriv::BlenderPrivConfig> _priv_config;
};

extern SmartPtr<SoftHandler> create_soft_blender ();
}

#endif //XCAM_SOFT_BLENDER_H
