/*
 * soft_copy_task.cpp - soft copy implementation
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
 * Author: Yinhang Liu <yinhangx.liu@intel.com>
 */

#include "soft_copy_task.h"

namespace XCam {

namespace XCamSoftTasks {

template <typename ImageT>
static inline void copy_line (const ImageT *in, ImageT *out, const uint32_t y, const uint32_t size)
{
    const typename ImageT::Type *in_ptr = in->get_buf_ptr (0, y);
    typename ImageT::Type *out_ptr = out->get_buf_ptr (0, y);

    memcpy (out_ptr, in_ptr, size);
}

XCamReturn
XCamSoftTasks::CopyTask::work_range (const SmartPtr<Arguments> &base, const WorkRange &range)
{
    SmartPtr<CopyTask::Args> args = base.dynamic_cast_ptr<CopyTask::Args> ();
    XCAM_ASSERT (args.ptr ());

    UcharImage *in_luma = args->in_luma.ptr (), *out_luma = args->out_luma.ptr ();
    Uchar2Image *in_uv = args->in_uv.ptr (), *out_uv = args->out_uv.ptr ();
    XCAM_ASSERT (in_luma && in_uv);
    XCAM_ASSERT (out_luma && out_uv);

    uint32_t luma_size = in_luma->get_width () * in_luma->pixel_size ();
    uint32_t uv_size = in_uv->get_width () * in_uv->pixel_size ();
    for (uint32_t y = range.pos[1]; y < range.pos[1] + range.pos_len[1]; ++y) {
        uint32_t luma_y = y * 2;
        copy_line<UcharImage> (in_luma, out_luma, luma_y, luma_size);
        copy_line<UcharImage> (in_luma, out_luma, luma_y + 1, luma_size);

        uint32_t uv_y = y;
        copy_line<Uchar2Image> (in_uv, out_uv, uv_y, uv_size);
    }

    XCAM_LOG_DEBUG ("CopyTask work on range:[x:%d, width:%d, y:%d, height:%d]",
                    range.pos[0], range.pos_len[0], range.pos[1], range.pos_len[1]);

    return XCAM_RETURN_NO_ERROR;
}

}

}
