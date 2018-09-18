/*
 * gl_blender.cpp - gl blender implementation
 *
 *  Copyright (c) 2018 Intel Corporation
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

#include "xcam_utils.h"
#include "image_file_handle.h"
#include "gl_utils.h"
#include "gl_video_buffer.h"
#include "gl_blender_shaders_priv.h"
#include "gl_blender.h"
#include <map>

#define OVERLAP_POOL_SIZE 6
#define LAP_POOL_SIZE 4

#define GAUSS_RADIUS 2
#define GAUSS_DIAMETER  ((GAUSS_RADIUS)*2+1)

const float gauss_coeffs[GAUSS_DIAMETER] = {0.152f, 0.222f, 0.252f, 0.222f, 0.152f};

#define DUMP_BUFFER 0

#define CHECK_RET(ret, format, ...) \
    if ((ret) < XCAM_RETURN_NO_ERROR) {          \
        XCAM_LOG_ERROR (format, ## __VA_ARGS__); \
    }

namespace XCam {

using namespace XCamGLShaders;

DECLARE_WORK_CALLBACK (CbGaussScalePyr, GLBlender, gauss_scale_done);
DECLARE_WORK_CALLBACK (CbLapTransPyr, GLBlender, lap_trans_done);
DECLARE_WORK_CALLBACK (CbBlendPyr, GLBlender, blend_done);
DECLARE_WORK_CALLBACK (CbReconstructPyr, GLBlender, reconstruct_done);

typedef std::map<void*, SmartPtr<GLBlendPyrShader::Args>> MapBlendArgs;
typedef std::map<void*, SmartPtr<GLReconstructPyrShader::Args>> MapReconstructArgs;

namespace GLBlenderPriv {

struct PyramidResource {
    SmartPtr<BufferPool>                overlap_pool;
    SmartPtr<GLGaussScalePyrShader>     gauss_scale[GLBlender::BufIdxCount];
    SmartPtr<GLLapTransPyrShader>       lap_trans[GLBlender::BufIdxCount];
    SmartPtr<GLReconstructPyrShader>    reconstruct;
    SmartPtr<GLBuffer>                  coef_mask;
    MapReconstructArgs                  reconstruct_args;
};

class BlenderPrivConfig {
public:
    PyramidResource               pyr_layer[XCAM_GL_PYRAMID_MAX_LEVEL];
    uint32_t                      pyr_levels;

    SmartPtr<GLBlendPyrShader>    top_level_blend;
    SmartPtr<BufferPool>          first_lap_pool;
    SmartPtr<GLBuffer>            first_mask;

    Mutex                         map_args_mutex;
    MapBlendArgs                  blend_args;

private:
    GLBlender                    *_blender;

public:
    BlenderPrivConfig (GLBlender *blender, uint32_t level)
        : pyr_levels (level)
        , _blender (blender)
    {}

    XCamReturn init_first_masks (uint32_t width, uint32_t height);
    XCamReturn scale_down_masks (uint32_t level, uint32_t width, uint32_t height);

    XCamReturn start_gauss_scale (
        const SmartPtr<ImageHandler::Parameters> &param,
        const SmartPtr<VideoBuffer> &in_buf,
        uint32_t level, GLBlender::BufIdx idx);

    XCamReturn start_lap_trans (
        const SmartPtr<ImageHandler::Parameters> &param,
        const SmartPtr<GLGaussScalePyrShader::Args> &gauss_scale_args,
        uint32_t level, GLBlender::BufIdx idx);

    XCamReturn start_blend (
        const SmartPtr<ImageHandler::Parameters> &param,
        const SmartPtr<VideoBuffer> &buf, GLBlender::BufIdx idx);

    XCamReturn start_reconstruct_by_lap (
        const SmartPtr<ImageHandler::Parameters> &param,
        const SmartPtr<VideoBuffer> &lap,
        uint32_t level, GLBlender::BufIdx idx);
    XCamReturn start_reconstruct_by_gauss (
        const SmartPtr<ImageHandler::Parameters> &param,
        const SmartPtr<VideoBuffer> &prev_blend_buf, uint32_t level);
    XCamReturn start_reconstruct (const SmartPtr<GLReconstructPyrShader::Args> &args, uint32_t level);
    XCamReturn stop ();
};

};

#if DUMP_BUFFER
#define dump_buf dump_buf_perfix_path

static void
dump_level_buf (const SmartPtr<VideoBuffer> &buf, const char *name, uint32_t level, uint32_t idx)
{
    XCAM_ASSERT (name);

    char file_name[256];
    snprintf (file_name, 256, "%s-L%d-Idx%d", name, level, idx);
    dump_buf_perfix_path (buf, file_name);
}
#else
static void
dump_level_buf (const SmartPtr<VideoBuffer> &buf, ...) {
    XCAM_UNUSED (buf);
}

static void
dump_buf (const SmartPtr<VideoBuffer> &buf, ...) {
    XCAM_UNUSED (buf);
}
#endif

GLBlender::GLBlender (const char *name)
    : GLImageHandler (name)
    , Blender (GL_BLENDER_ALIGN_X, GL_BLENDER_ALIGN_Y)
{
    SmartPtr<GLBlenderPriv::BlenderPrivConfig> config =
        new GLBlenderPriv::BlenderPrivConfig (this, XCAM_GL_PYRAMID_DEFAULT_LEVEL);
    XCAM_ASSERT (config.ptr ());
    _priv_config = config;
}

GLBlender::~GLBlender ()
{
}

XCamReturn
GLBlender::terminate ()
{
    _priv_config->stop ();
    return GLImageHandler::terminate ();
}

XCamReturn
GLBlender::blend (
    const SmartPtr<VideoBuffer> &in0,
    const SmartPtr<VideoBuffer> &in1,
    SmartPtr<VideoBuffer> &out_buf)
{
    XCAM_ASSERT (in0.ptr () && in1.ptr ());

    SmartPtr<BlenderParam> param = new BlenderParam (in0, in1, out_buf);
    XCAM_ASSERT (param.ptr ());

    XCamReturn ret = execute_buffer (param, true);
    if (xcam_ret_is_ok (ret) && !out_buf.ptr ()) {
        out_buf = param->out_buf;
    }

    return ret;
}

XCamReturn
GLBlenderPriv::BlenderPrivConfig::stop ()
{
    for (uint32_t i = 0; i < pyr_levels; ++i) {
        pyr_layer[i].gauss_scale[GLBlender::Idx0].release ();
        pyr_layer[i].gauss_scale[GLBlender::Idx1].release ();
        pyr_layer[i].lap_trans[GLBlender::Idx0].release ();
        pyr_layer[i].lap_trans[GLBlender::Idx1].release ();
        pyr_layer[i].reconstruct.release ();

        if (pyr_layer[i].overlap_pool.ptr ()) {
            pyr_layer[i].overlap_pool->stop ();
        }
    }

    top_level_blend.release ();
    if (first_lap_pool.ptr ()) {
        first_lap_pool->stop ();
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
GLBlenderPriv::BlenderPrivConfig::init_first_masks (uint32_t width, uint32_t height)
{
    XCAM_ASSERT (!first_mask.ptr ());
    XCAM_ASSERT (width && (width % GL_BLENDER_ALIGN_X == 0));
    XCAM_FAIL_RETURN (
        ERROR, height == 1, XCAM_RETURN_ERROR_PARAM,
        "blender(%s) mask buffer only supports one-dimensional array", XCAM_STR (_blender->get_name ()));

    uint32_t buf_size = width * sizeof (uint8_t);
    SmartPtr<GLBuffer> buf = GLBuffer::create_buffer (GL_SHADER_STORAGE_BUFFER, NULL, buf_size);
    XCAM_ASSERT (buf.ptr ());

    GLBufferDesc desc;
    desc.width = width;
    desc.height = 1;
    desc.size = buf_size;
    buf->set_buffer_desc (desc);

    std::vector<float> gauss_table;
    uint32_t quater = width / 4;
    XCAM_ASSERT (quater > 1);

    get_gauss_table (quater, (quater + 1) / 4.0f, gauss_table, false);
    for (uint32_t i = 0; i < gauss_table.size (); ++i) {
        float value = ((i < quater) ? (128.0f * (2.0f - gauss_table[i])) : (128.0f * gauss_table[i]));
        value = XCAM_CLAMP (value, 0.0f, 255.0f);
        gauss_table[i] = value;
    }

    uint8_t *mask_ptr = (uint8_t *) buf->map_range (0, buf_size, GL_MAP_WRITE_BIT);
    XCAM_FAIL_RETURN (ERROR, mask_ptr, XCAM_RETURN_ERROR_PARAM, "map range failed");

    uint32_t gauss_start_pos = (width - gauss_table.size ()) / 2;
    uint32_t idx = 0;
    for (idx = 0; idx < gauss_start_pos; ++idx) {
        mask_ptr[idx] = 255;
    }
    for (uint32_t i = 0; i < gauss_table.size (); ++idx, ++i) {
        mask_ptr[idx] = (uint8_t) gauss_table[i];
    }
    for (; idx < width; ++idx) {
        mask_ptr[idx] = 0;
    }
    buf->unmap ();

    first_mask = buf;

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
GLBlenderPriv::BlenderPrivConfig::scale_down_masks (uint32_t level, uint32_t width, uint32_t height)
{
    XCAM_ASSERT (width && (width % GL_BLENDER_ALIGN_X == 0));
    XCAM_FAIL_RETURN (
        ERROR, height == 1, XCAM_RETURN_ERROR_PARAM,
        "blender(%s) mask buffer only supports one-dimensional array", XCAM_STR (_blender->get_name ()));

    uint32_t buf_size = width * sizeof (uint8_t);
    SmartPtr<GLBuffer> buf = GLBuffer::create_buffer (GL_SHADER_STORAGE_BUFFER, NULL, buf_size);
    XCAM_ASSERT (buf.ptr ());

    GLBufferDesc desc;
    desc.width = width;
    desc.height = 1;
    desc.size = buf_size;
    buf->set_buffer_desc (desc);

    SmartPtr<GLBuffer> prev_mask;
    if (level == 0) {
        prev_mask = first_mask;
    } else {
        prev_mask = pyr_layer[level - 1].coef_mask;
    }
    XCAM_ASSERT (prev_mask.ptr ());

    const GLBufferDesc prev_desc = prev_mask->get_buffer_desc ();
    uint8_t *prev_ptr = (uint8_t *) prev_mask->map_range (0, prev_desc.size, GL_MAP_READ_BIT);
    XCAM_FAIL_RETURN (ERROR, prev_ptr, XCAM_RETURN_ERROR_PARAM, "map range failed");

    uint8_t *cur_ptr = (uint8_t *) buf->map_range (0, desc.size, GL_MAP_WRITE_BIT);
    XCAM_FAIL_RETURN (ERROR, cur_ptr, XCAM_RETURN_ERROR_PARAM, "map range failed");

    for (uint32_t i = 0; i < desc.width; ++i) {
        int prev_start = i * 2 - 2;
        float sum = 0.0f;

        for (int j = 0; j < GAUSS_DIAMETER; ++j) {
            int prev_idx = XCAM_CLAMP (prev_start + j, 0, (int)prev_desc.width);
            sum += prev_ptr[prev_idx] * gauss_coeffs[j];
        }

        cur_ptr[i] = XCAM_CLAMP (sum, 0.0f, 255.0f);
    }

    buf->unmap ();
    prev_mask->unmap ();

    pyr_layer[level].coef_mask = buf;

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
GLBlenderPriv::BlenderPrivConfig::start_gauss_scale (
    const SmartPtr<ImageHandler::Parameters> &param,
    const SmartPtr<VideoBuffer> &in_buf,
    uint32_t level, GLBlender::BufIdx idx)
{
    XCAM_ASSERT (in_buf.ptr ());
    XCAM_ASSERT (level < pyr_levels);
    XCAM_ASSERT (idx < GLBlender::BufIdxCount);
    XCAM_ASSERT (pyr_layer[level].gauss_scale[idx].ptr ());
    XCAM_ASSERT (pyr_layer[level].overlap_pool.ptr ());

    SmartPtr<VideoBuffer> out_buf = pyr_layer[level].overlap_pool->get_buffer ();
    XCAM_FAIL_RETURN (
        ERROR, out_buf.ptr (), XCAM_RETURN_ERROR_MEM,
        "blender(%s) start_gauss_scale failed, output buffer is empty, level:%d, idx:%d",
        XCAM_STR (_blender->get_name ()), level, (int)idx);

    SmartPtr<GLGaussScalePyrShader::Args> args = new GLGaussScalePyrShader::Args (param, level, idx);
    XCAM_ASSERT (args.ptr ());
    args->in_glbuf = get_glbuffer (in_buf);
    args->out_glbuf = get_glbuffer (out_buf);
    args->out_video_buf = out_buf;

    if (level == 0) {
        const Rect area = _blender->get_input_merge_area (idx);
        XCAM_FAIL_RETURN (
            ERROR, 
            area.pos_y == 0 && area.width && area.height &&
            area.pos_x % GL_BLENDER_ALIGN_X == 0 &&
            area.width % GL_BLENDER_ALIGN_X == 0 &&
            area.height % GL_BLENDER_ALIGN_Y == 0,
            XCAM_RETURN_ERROR_PARAM,
            "blender(%s) invalid input merge area, pos_x:%d, pos_y:%d, width:%d, height:%d, level:%d, idx:%d",
            XCAM_STR (_blender->get_name ()), area.pos_x, area.pos_y, area.width, area.height, level, (int)idx);

        args->merge_area = area;
    } else {
        const VideoBufferInfo &info = in_buf->get_video_info ();
        XCAM_FAIL_RETURN (
            ERROR, 
            info.width && info.height &&
            info.width % GL_BLENDER_ALIGN_X == 0 &&
            info.height % GL_BLENDER_ALIGN_Y == 0,
            XCAM_RETURN_ERROR_PARAM,
            "blender(%s) invalid buffer info, width:%d, height:%d, level:%d, idx:%d",
            XCAM_STR (_blender->get_name ()), info.width, info.height, level, (int)idx);

        args->merge_area = Rect (0, 0, info.width, info.height);
    }

    return pyr_layer[level].gauss_scale[idx]->work (args);
}

XCamReturn
GLBlenderPriv::BlenderPrivConfig::start_lap_trans (
    const SmartPtr<ImageHandler::Parameters> &param,
    const SmartPtr<GLGaussScalePyrShader::Args> &gauss_scale_args,
    uint32_t level, GLBlender::BufIdx idx)
{
    XCAM_ASSERT (level < pyr_levels);
    XCAM_ASSERT (pyr_layer[level].lap_trans[idx].ptr ());
    XCAM_ASSERT (idx < GLBlender::BufIdxCount);

    SmartPtr<VideoBuffer> out_buf;
    if (level == 0) {
        XCAM_ASSERT (first_lap_pool.ptr ());
        out_buf = first_lap_pool->get_buffer ();
    } else {
        XCAM_ASSERT (pyr_layer[level - 1].overlap_pool.ptr ());
        out_buf = pyr_layer[level - 1].overlap_pool->get_buffer ();
    }
    XCAM_FAIL_RETURN (
        ERROR, out_buf.ptr (), XCAM_RETURN_ERROR_MEM,
        "blender(%s) start_lap_trans failed, output buffer is empty, level:%d, idx:%d",
        XCAM_STR (_blender->get_name ()), level, (int)idx);

    SmartPtr<GLLapTransPyrShader::Args> args = new GLLapTransPyrShader::Args (param, level, idx);
    XCAM_ASSERT (args.ptr ());
    args->in_glbuf = gauss_scale_args->in_glbuf;
    args->gaussscale_glbuf = gauss_scale_args->out_glbuf;
    args->out_glbuf = get_glbuffer (out_buf);
    args->merge_area = gauss_scale_args->merge_area;
    args->out_video_buf = out_buf;

    return pyr_layer[level].lap_trans[idx]->work (args);
}

XCamReturn
GLBlenderPriv::BlenderPrivConfig::start_blend (
    const SmartPtr<ImageHandler::Parameters> &param,
    const SmartPtr<VideoBuffer> &buf, GLBlender::BufIdx idx)
{
    XCAM_ASSERT (idx < GLBlender::BufIdxCount);
    XCAM_ASSERT (top_level_blend.ptr ());

    uint32_t top_level = pyr_levels - 1;
    XCAM_ASSERT (pyr_layer[top_level].overlap_pool.ptr ());
    XCAM_ASSERT (pyr_layer[top_level].coef_mask.ptr ());

    SmartPtr<GLBlendPyrShader::Args> args;
    {
        SmartLock locker (map_args_mutex);
        MapBlendArgs::iterator i = blend_args.find (param.ptr ());
        if (i == blend_args.end ()) {
            args = new GLBlendPyrShader::Args (param);
            XCAM_ASSERT (args.ptr ());
            blend_args.insert (std::make_pair((void*)param.ptr (), args));
            XCAM_LOG_DEBUG ("blender(%s) init blend args, idx:%d", XCAM_STR (_blender->get_name ()), (int)idx);
        } else {
            args = (*i).second;
        }

        if (idx == GLBlender::Idx0) {
            args->in0_glbuf = get_glbuffer (buf);
        } else {
            args->in1_glbuf = get_glbuffer (buf);
        }

        if (!args->in0_glbuf.ptr () || !args->in1_glbuf.ptr ())
            return XCAM_RETURN_BYPASS;

        blend_args.erase (i);
    }

    args->mask_glbuf = pyr_layer[top_level].coef_mask;

    SmartPtr<VideoBuffer> out_buf = pyr_layer[top_level].overlap_pool->get_buffer ();
    XCAM_FAIL_RETURN (
        ERROR, out_buf.ptr (), XCAM_RETURN_ERROR_MEM,
        "blender(%s) start_blend failed, output buffer is empty, idx:%d",
        XCAM_STR (_blender->get_name ()), (int)idx);
    args->out_glbuf = get_glbuffer (out_buf);
    args->out_video_buf = out_buf;

    return top_level_blend->work (args);
}

XCamReturn
GLBlenderPriv::BlenderPrivConfig::start_reconstruct (
    const SmartPtr<GLReconstructPyrShader::Args> &args, uint32_t level)
{
    XCAM_ASSERT (args.ptr ());
    XCAM_ASSERT (level < pyr_levels);
    XCAM_ASSERT (pyr_layer[level].reconstruct.ptr ());
    XCAM_ASSERT (args->lap0_glbuf.ptr () && args->lap1_glbuf.ptr () && args->prev_blend_glbuf.ptr ());

    SmartPtr<VideoBuffer> out_buf;
    if (level == 0) {
        const SmartPtr<ImageHandler::Parameters> param = args->get_param ();
        XCAM_ASSERT (param.ptr () && param->out_buf.ptr ());
        out_buf = param->out_buf;

        XCAM_ASSERT (first_mask.ptr ());
        args->mask_glbuf = first_mask;

        const Rect area = _blender->get_merge_window ();
        XCAM_FAIL_RETURN (
            ERROR, 
            area.pos_y == 0 && area.width && area.height &&
            area.pos_x % GL_BLENDER_ALIGN_X == 0 &&
            area.width % GL_BLENDER_ALIGN_X == 0 &&
            area.height % GL_BLENDER_ALIGN_Y == 0,
            XCAM_RETURN_ERROR_PARAM,
            "blender(%s) invalid output merge area, pos_x:%d, pos_y:%d, width:%d, height:%d, level:%d",
            XCAM_STR (_blender->get_name ()), area.pos_x, area.pos_y, area.width, area.height, level);

        args->merge_area = area;
    } else {
        out_buf = pyr_layer[level - 1].overlap_pool->get_buffer ();
        XCAM_FAIL_RETURN (
            ERROR, out_buf.ptr (), XCAM_RETURN_ERROR_MEM,
            "blender(%s) start_reconstruct failed, out buffer is empty, level:%d",
            XCAM_STR (_blender->get_name ()), level);

        XCAM_ASSERT (pyr_layer[level - 1].coef_mask.ptr ());
        args->mask_glbuf = pyr_layer[level - 1].coef_mask;

        const VideoBufferInfo &info = out_buf->get_video_info ();
        XCAM_FAIL_RETURN (
            ERROR, 
            info.width && info.height &&
            info.width % GL_BLENDER_ALIGN_X == 0 &&
            info.height % GL_BLENDER_ALIGN_Y == 0,
            XCAM_RETURN_ERROR_PARAM,
            "blender(%s) invalid buffer info, width:%d, height:%d, level:%d",
            XCAM_STR (_blender->get_name ()), info.width, info.height, level);

        args->merge_area = Rect (0, 0, info.width, info.height);
    }
    args->out_glbuf = get_glbuffer (out_buf);
    args->out_video_buf = out_buf;

    return pyr_layer[level].reconstruct->work (args);
}

XCamReturn
GLBlenderPriv::BlenderPrivConfig::start_reconstruct_by_gauss (
    const SmartPtr<ImageHandler::Parameters> &param,
    const SmartPtr<VideoBuffer> &prev_blend_buf, uint32_t level)
{
    XCAM_ASSERT (prev_blend_buf.ptr ());
    XCAM_ASSERT (level < pyr_levels);

    SmartPtr<GLReconstructPyrShader::Args> args;
    {
        SmartLock locker (map_args_mutex);
        MapReconstructArgs::iterator i = pyr_layer[level].reconstruct_args.find (param.ptr ());
        if (i == pyr_layer[level].reconstruct_args.end ()) {
            args = new GLReconstructPyrShader::Args (param, level);
            XCAM_ASSERT (args.ptr ());
            pyr_layer[level].reconstruct_args.insert (std::make_pair((void*)param.ptr (), args));
            XCAM_LOG_DEBUG ("blender(%s) init reconstruct_args, level:%d", XCAM_STR (_blender->get_name ()), level);
        } else {
            args = (*i).second;
        }

        args->prev_blend_glbuf = get_glbuffer (prev_blend_buf);

        if (!args->lap0_glbuf.ptr () || !args->lap1_glbuf.ptr ())
            return XCAM_RETURN_BYPASS;

        pyr_layer[level].reconstruct_args.erase (i);
    }

    return start_reconstruct (args, level);
}

XCamReturn
GLBlenderPriv::BlenderPrivConfig::start_reconstruct_by_lap (
    const SmartPtr<ImageHandler::Parameters> &param,
    const SmartPtr<VideoBuffer> &lap,
    uint32_t level, GLBlender::BufIdx idx)
{
    XCAM_ASSERT (lap.ptr ());
    XCAM_ASSERT (level < pyr_levels);
    XCAM_ASSERT (idx < GLBlender::BufIdxCount);

    SmartPtr<GLReconstructPyrShader::Args> args;
    {
        SmartLock locker (map_args_mutex);
        MapReconstructArgs::iterator i = pyr_layer[level].reconstruct_args.find (param.ptr ());
        if (i == pyr_layer[level].reconstruct_args.end ()) {
            args = new GLReconstructPyrShader::Args (param, level);
            XCAM_ASSERT (args.ptr ());
            pyr_layer[level].reconstruct_args.insert (std::make_pair((void*)param.ptr (), args));
            XCAM_LOG_DEBUG ("blender(%s) init reconstruct_args, level:%d", XCAM_STR (_blender->get_name ()), level);
        } else {
            args = (*i).second;
        }

        if (idx == GLBlender::Idx0)
            args->lap0_glbuf = get_glbuffer (lap);
        else
            args->lap1_glbuf = get_glbuffer (lap);

        if (!args->lap0_glbuf.ptr () || !args->lap1_glbuf.ptr () || !args->prev_blend_glbuf.ptr ())
            return XCAM_RETURN_BYPASS;

        pyr_layer[level].reconstruct_args.erase (i);
    }

    return start_reconstruct (args, level);
}

XCamReturn
GLBlender::start_work (const SmartPtr<ImageHandler::Parameters> &base)
{
    XCAM_ASSERT (base.ptr ());
    SmartPtr<BlenderParam> param = base.dynamic_cast_ptr<BlenderParam> ();
    XCAM_ASSERT (param.ptr ());
    XCAM_ASSERT (param->in_buf.ptr () && param->in1_buf.ptr () && param->out_buf.ptr ());

    dump_level_buf (param->in_buf, "input", 0, 0);
    dump_level_buf (param->in1_buf, "input", 0, 1);

    // start gauss scale level:0 idx:0
    XCamReturn ret = _priv_config->start_gauss_scale (param, param->in_buf, 0, GLBlender::Idx0);
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret,
        "blender(%s) start gauss scale failed, level:0 idx:0", XCAM_STR (get_name ()));

    // start gauss scale level:0 idx:1
    ret = _priv_config->start_gauss_scale (param, param->in1_buf, 0, GLBlender::Idx1);
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret,
        "blender(%s) start gauss scale failed, level:0 idx:1", XCAM_STR (get_name ()));

    return ret;
};

XCamReturn
GLBlender::configure_resource (const SmartPtr<Parameters> &param)
{
    XCAM_ASSERT (param.ptr () && param->in_buf.ptr ());
    XCAM_ASSERT (_priv_config->pyr_levels <= XCAM_GL_PYRAMID_MAX_LEVEL);

    const VideoBufferInfo &in0_info = param->in_buf->get_video_info ();
    XCAM_FAIL_RETURN (
        ERROR, in0_info.format == V4L2_PIX_FMT_NV12, XCAM_RETURN_ERROR_PARAM,
        "blender(%s) only support NV12 format, but input format is %s",
        XCAM_STR(get_name ()), xcam_fourcc_to_string (in0_info.format));

    Rect in0_area, in1_area, out_area;
    in0_area = get_input_merge_area (Idx0);
    in1_area = get_input_merge_area (Idx1);
    out_area = get_merge_window ();
    XCAM_FAIL_RETURN (
        ERROR,
        in0_area.width && in0_area.height &&
        in0_area.width == in1_area.width && in0_area.height == in1_area.height &&
        in0_area.width == out_area.width && in0_area.height == out_area.height,
        XCAM_RETURN_ERROR_PARAM,
        "blender(%s) invalid input/output overlap area, input0:%dx%d, input1:%dx%d, output:%dx%d",
        XCAM_STR(get_name ()), in0_area.width, in0_area.height,
        in1_area.width, in1_area.height, out_area.width, out_area.height);

    VideoBufferInfo out_info;
    uint32_t out_width, out_height;
    get_output_size (out_width, out_height);
    XCAM_FAIL_RETURN (
        ERROR, out_width && out_height, XCAM_RETURN_ERROR_PARAM,
        "blender(%s) invalid output size, output size:%dx%d",
        XCAM_STR(get_name ()), out_width, out_height);

    out_info.init (
        in0_info.format, out_width, out_height,
        XCAM_ALIGN_UP (out_width, GL_BLENDER_ALIGN_X), XCAM_ALIGN_UP (out_height, GL_BLENDER_ALIGN_Y));
    set_out_video_info (out_info);

    VideoBufferInfo overlap_info;
    Rect merge_size = get_merge_window ();
    XCAM_FAIL_RETURN (
            ERROR, 
            merge_size.width && merge_size.height &&
            merge_size.width % GL_BLENDER_ALIGN_X == 0 &&
            merge_size.height % GL_BLENDER_ALIGN_Y == 0,
            XCAM_RETURN_ERROR_PARAM,
            "blender(%s) invalid merge size, width:%d, height:%d",
            XCAM_STR (get_name ()), merge_size.width, merge_size.height);

    overlap_info.init (in0_info.format, merge_size.width, merge_size.height);
    SmartPtr<BufferPool> first_lap_pool = new GLVideoBufferPool (overlap_info);
    XCAM_ASSERT (first_lap_pool.ptr ());
    XCAM_FAIL_RETURN (
        ERROR, first_lap_pool->reserve (LAP_POOL_SIZE), XCAM_RETURN_ERROR_MEM,
        "blender(%s) reserve lap buffer pool failed, overlap size:%dx%d",
        XCAM_STR(get_name ()), overlap_info.width, overlap_info.height);
    _priv_config->first_lap_pool = first_lap_pool;

    SmartPtr<Worker::Callback> gauss_scale_cb = new CbGaussScalePyr (this);
    SmartPtr<Worker::Callback> lap_trans_cb = new CbLapTransPyr (this);
    SmartPtr<Worker::Callback> reconstruct_cb = new CbReconstructPyr (this);
    XCAM_ASSERT (gauss_scale_cb.ptr () && lap_trans_cb.ptr () && reconstruct_cb.ptr ());

    XCamReturn ret = _priv_config->init_first_masks (merge_size.width, 1);
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret,
        "blender(%s) init first masks failed", XCAM_STR (get_name ()));

    for (uint32_t i = 0; i < _priv_config->pyr_levels; ++i) {
        merge_size.width = XCAM_ALIGN_UP ((merge_size.width + 1) / 2, GL_BLENDER_ALIGN_X);
        merge_size.height = XCAM_ALIGN_UP ((merge_size.height + 1) / 2, GL_BLENDER_ALIGN_Y);
        overlap_info.init (in0_info.format, merge_size.width, merge_size.height);

        SmartPtr<BufferPool> pool = new GLVideoBufferPool (overlap_info);
        XCAM_ASSERT (pool.ptr ());
        XCAM_FAIL_RETURN (
            ERROR, pool->reserve (OVERLAP_POOL_SIZE), XCAM_RETURN_ERROR_MEM,
            "blender(%s) reserve buffer pool failed, overlap size:%dx%d",
            XCAM_STR(get_name ()), overlap_info.width, overlap_info.height);
        _priv_config->pyr_layer[i].overlap_pool = pool;

        ret = _priv_config->scale_down_masks (i, merge_size.width, 1);
        XCAM_FAIL_RETURN (
            ERROR, xcam_ret_is_ok (ret), ret,
            "blender(%s) scale down masks failed, level:%d", XCAM_STR (get_name ()), i);

        _priv_config->pyr_layer[i].gauss_scale[GLBlender::Idx0] = create_gauss_scale_pyr_shader (gauss_scale_cb);
        XCAM_ASSERT (_priv_config->pyr_layer[i].gauss_scale[GLBlender::Idx0].ptr ());
        _priv_config->pyr_layer[i].gauss_scale[GLBlender::Idx1] = create_gauss_scale_pyr_shader (gauss_scale_cb);
        XCAM_ASSERT (_priv_config->pyr_layer[i].gauss_scale[GLBlender::Idx1].ptr ());
        _priv_config->pyr_layer[i].lap_trans[GLBlender::Idx0] = create_lap_trans_pyr_shader (lap_trans_cb);
        XCAM_ASSERT (_priv_config->pyr_layer[i].lap_trans[GLBlender::Idx0].ptr ());
        _priv_config->pyr_layer[i].lap_trans[GLBlender::Idx1] = create_lap_trans_pyr_shader (lap_trans_cb);
        XCAM_ASSERT (_priv_config->pyr_layer[i].lap_trans[GLBlender::Idx1].ptr ());
        _priv_config->pyr_layer[i].reconstruct = create_reconstruct_pyr_shader (reconstruct_cb);
        XCAM_ASSERT (_priv_config->pyr_layer[i].reconstruct.ptr ());
    }

    SmartPtr<Worker::Callback> blend_cb = new CbBlendPyr (this);
    XCAM_ASSERT (blend_cb.ptr ());
    _priv_config->top_level_blend = create_blend_pyr_shader (blend_cb);
    XCAM_ASSERT (_priv_config->top_level_blend.ptr ());

    return XCAM_RETURN_NO_ERROR;
}

void
GLBlender::gauss_scale_done (
    const SmartPtr<Worker> &worker, const SmartPtr<Worker::Arguments> &base, const XCamReturn error)
{
    XCAM_UNUSED (worker);
    XCAM_ASSERT (base.ptr ());

    SmartPtr<GLGaussScalePyrShader::Args> args = base.dynamic_cast_ptr<GLGaussScalePyrShader::Args> ();
    XCAM_ASSERT (args.ptr ());
    uint32_t level = args->level;
    XCAM_ASSERT (level < _priv_config->pyr_levels);
    uint32_t next_level = level + 1;
    BufIdx idx = args->idx;

    const SmartPtr<ImageHandler::Parameters> param = args->get_param ();
    XCAM_ASSERT (param.ptr ());

    dump_level_buf (args->out_video_buf, "gauss-scale", level, idx);

    XCamReturn ret = _priv_config->start_lap_trans (param, args, level, idx);
    CHECK_RET (ret, "execute laplace transformation failed, level:%d idx:%d", level, idx);

    if (next_level == _priv_config->pyr_levels) { // top level
        ret = _priv_config->start_blend (param, args->out_video_buf, idx);
        CHECK_RET (ret, "execute blend failed, level:%d idx:%d", next_level, idx);
    } else {
        ret = _priv_config->start_gauss_scale (param, args->out_video_buf, next_level, idx);
        CHECK_RET (ret, "execute gauss scale failed, level:%d idx:%d", next_level, idx);
    }

    execute_done (param, error);
}

void
GLBlender::lap_trans_done (
    const SmartPtr<Worker> &worker, const SmartPtr<Worker::Arguments> &base, const XCamReturn error)
{
    XCAM_UNUSED (worker);
    XCAM_ASSERT (base.ptr ());

    SmartPtr<GLLapTransPyrShader::Args> args = base.dynamic_cast_ptr<GLLapTransPyrShader::Args> ();
    XCAM_ASSERT (args.ptr ());
    uint32_t level = args->level;
    XCAM_ASSERT (level < _priv_config->pyr_levels);
    BufIdx idx = args->idx;

    const SmartPtr<ImageHandler::Parameters> param = args->get_param ();
    XCAM_ASSERT (param.ptr ());

    dump_level_buf (args->out_video_buf, "lap", level, idx);

    XCamReturn ret = _priv_config->start_reconstruct_by_lap (param, args->out_video_buf, level, idx);
    CHECK_RET (ret, "execute reconstruct by lap failed, level:%d idx:%d", level, idx);

    execute_done (param, error);
}

void
GLBlender::blend_done (
    const SmartPtr<Worker> &worker, const SmartPtr<Worker::Arguments> &base, const XCamReturn error)
{
    XCAM_UNUSED (worker);
    XCAM_ASSERT (base.ptr ());

    SmartPtr<GLBlendPyrShader::Args> args = base.dynamic_cast_ptr<GLBlendPyrShader::Args> ();
    XCAM_ASSERT (args.ptr ());
    const SmartPtr<ImageHandler::Parameters> param = args->get_param ();
    XCAM_ASSERT (param.ptr ());

    dump_buf (args->out_video_buf, "blend-top");

    XCamReturn ret = _priv_config->start_reconstruct_by_gauss (param, args->out_video_buf, _priv_config->pyr_levels - 1);
    CHECK_RET (ret, "execute reconstruct by gauss failed, level:%d", _priv_config->pyr_levels - 1);

    execute_done (param, error);
}

void
GLBlender::reconstruct_done (
    const SmartPtr<Worker> &worker, const SmartPtr<Worker::Arguments> &base, const XCamReturn error)
{
    XCAM_UNUSED (worker);
    XCAM_ASSERT (base.ptr ());

    SmartPtr<GLReconstructPyrShader::Args> args = base.dynamic_cast_ptr<GLReconstructPyrShader::Args> ();
    XCAM_ASSERT (args.ptr ());
    uint32_t level = args->level;
    XCAM_ASSERT (level < _priv_config->pyr_levels);

    const SmartPtr<ImageHandler::Parameters> param = args->get_param ();
    XCAM_ASSERT (param.ptr ());

    dump_level_buf (args->out_video_buf, "reconstruct", level, 0);

    if (level == 0) {
        execute_done (param, error);
        return;
    }

    XCamReturn ret = _priv_config->start_reconstruct_by_gauss (param, args->out_video_buf, level - 1);
    CHECK_RET (ret, "execute reconstruct by gauss failed, level:%d", level - 1);

    execute_done (param, error);
}

SmartPtr<GLImageHandler>
create_gl_blender ()
{
    SmartPtr<GLBlender> blender = new GLBlender();
    XCAM_ASSERT (blender.ptr ());
    return blender;
}

SmartPtr<Blender>
Blender::create_gl_blender ()
{
    SmartPtr<GLImageHandler> handler = XCam::create_gl_blender ();
    return handler.dynamic_cast_ptr<Blender> ();
}

}
