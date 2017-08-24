/*
 * stitcher.h - stitcher interface
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
 * Author: Yinhang Liu <yinhangx.liu@intel.com>
 */

#ifndef XCAM_INTERFACE_STITCHER_H
#define XCAM_INTERFACE_STITCHER_H

#include "xcam_utils.h"
#include "interface/data_types.h"
#include "video_buffer.h"

#define XCAM_STITCH_FISHEYE_MAX_NUM    6

namespace XCam {

enum StitchResMode {
    StitchRes1080P,
    StitchRes1080P4,
    StitchRes4K
};

struct StitchInfo {
    uint32_t merge_width[XCAM_STITCH_FISHEYE_MAX_NUM];

    ImageCropInfo crop[XCAM_STITCH_FISHEYE_MAX_NUM];
    FisheyeInfo fisheye_info[XCAM_STITCH_FISHEYE_MAX_NUM];

    StitchInfo () {
        xcam_mem_clear (merge_width);
    }
};

struct ImageMergeInfo {
    Rect left;
    Rect right;
};

class Stitcher;

class Stitcher
{
public:
    explicit Stitcher (StitchResMode res_mode);
    virtual ~Stitcher ();
    static SmartPtr<Stitcher> create_ocl_stitcher ();
    static SmartPtr<Stitcher> create_soft_stitcher ();

    bool set_stitch_info (const StitchInfo &stitch_info);
    StitchInfo get_stitch_info ();
    void set_output_size (uint32_t width, uint32_t height) {
        _output_width = width; //XCAM_ALIGN_UP (width, XCAM_BLENDER_ALIGNED_WIDTH);
        _output_height = height;
    }
    virtual XCamReturn stitch_buffers (const VideoBufferList &in_bufs, SmartPtr<VideoBuffer> &out_buf) = 0;

private:
    XCAM_DEAD_COPY (Stitcher);

private:
    StitchResMode               _res_mode;
    uint32_t                    _output_width;
    uint32_t                    _output_height;

    ImageMergeInfo              _img_merge_info[XCAM_STITCH_FISHEYE_MAX_NUM];
    Rect                        _overlaps[XCAM_STITCH_FISHEYE_MAX_NUM][2];   // 2=>Overlap0 and overlap1

    StitchInfo                  _stitch_info;
    bool                        _is_stitch_inited;
};

}

#endif //XCAM_INTERFACE_STITCHER_H