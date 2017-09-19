/*
 * soft_blender.cpp - soft blender class implementation
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

#include "soft_blender.h"
#include "soft_image.h"
#include "soft_worker.h"
#include "soft_blender_tasks_priv.h"
#include "image_file_handle.h"
#include "soft_video_buf_allocator.h"
#include <map>

#define OVERLAP_POOL_SIZE 6
#define LAP_POOL_SIZE 4

#define DUMP_BLENDER 1

namespace XCam {

DECLARE_WORK_CALLBACK (CbGaussDownScale, SoftBlender, gauss_scale_done);
DECLARE_WORK_CALLBACK (CbBlendTask, SoftBlender, blend_task_done);
DECLARE_WORK_CALLBACK (CbReconstructTask, SoftBlender, reconstruct_done);
DECLARE_WORK_CALLBACK (CbLapTask, SoftBlender, lap_done);

typedef std::map<void*, SmartPtr<BlendTask::Args>> MapBlendArgs;
typedef std::map<void*, SmartPtr<ReconstructTask::Args>> MapReconsArgs;

namespace SoftBlenderPriv {

struct PyramidResource {
    SmartPtr<BufferPool>       overlap_pool;
    SmartPtr<GaussDownScale>   scale_task[SoftBlender::BufIdxCount];
    SmartPtr<LaplaceTask>      lap_task[SoftBlender::BufIdxCount];
    SmartPtr<ReconstructTask>  recon_task;
    SmartPtr<UcharImage>       coef_mask;
    MapReconsArgs              recons_args;
};

/* Level0: G[0] = gauss(in),  Lap[0] = in - upsample(G[0])
 Level1: G[1] = gauss(G[0]),  Lap[1] = G[0] - upsample(G[1])
..
 LevelN: G[N] = gauss(G[N-1]),
blend[N] = blend (Ga[N)], Gb[N)])
 Level(N-1): Reconst[N-1] = reconstruct (blend[N], LapA[N-1], LapB[N-1])
...
 Level1: reconst[1] = reconstruct (reconst[2], LapA[1], LapB[1])
 Level0: output = reconstruct (reconst[1], LapA[0], LapB[0])

 LevelN: Pool[N].size = G[N].size
 */
class BlenderPrivConfig {
public:
    PyramidResource        pyr_layer[XCAM_SOFT_PYRAMID_MAX_LEVEL];
    uint32_t               pyr_levels;
    SmartPtr<BlendTask>    last_level_blend;
    SmartPtr<BufferPool>   first_lap_pool;
    SmartPtr<UcharImage>   orig_mask;

    Mutex                  map_args_mutex;
    MapBlendArgs           blend_args;

private:
    SoftBlender           *_blender;

public:
    BlenderPrivConfig (SoftBlender *blender, uint32_t level)
        : pyr_levels (level)
        , _blender (blender)
    {}

    XCamReturn init_masks (uint32_t width, uint32_t height);

    XCamReturn start_scaler (
        const SmartPtr<ImageHandler::Parameters> &param,
        const SmartPtr<VideoBuffer> &in_buf,
        const uint32_t level, const SoftBlender::BufIdx idx);

    XCamReturn start_lap_task (
        const SmartPtr<ImageHandler::Parameters> &param,
        const uint32_t level, const SoftBlender::BufIdx idx,
        const SmartPtr<VideoBuffer> &orig, const SmartPtr<VideoBuffer> &gauss);
    XCamReturn start_blend_task (
        const SmartPtr<ImageHandler::Parameters> &param,
        const SmartPtr<VideoBuffer> &buf,
        const SoftBlender::BufIdx idx);

    XCamReturn start_reconstruct_task_by_lap (
        const SmartPtr<ImageHandler::Parameters> &param,
        const SmartPtr<VideoBuffer> &lap,
        const uint32_t level, const SoftBlender::BufIdx idx);
    XCamReturn start_reconstruct_task_by_gauss (
        const SmartPtr<ImageHandler::Parameters> &param,
        const SmartPtr<VideoBuffer> &gauss,
        const uint32_t level);
    XCamReturn start_reconstruct_task (const SmartPtr<ReconstructTask::Args> &args, const uint32_t level);
};

};

#if DUMP_BLENDER
static
void dump_buf (const SmartPtr<VideoBuffer> buf, const char *name)
{
    char file_name[256];
    XCAM_ASSERT (name);
    XCAM_ASSERT (buf.ptr ());

    const VideoBufferInfo &info = buf->get_video_info ();
    snprintf (file_name, 256, "%s-%dx%d.nv12", name, info.width, info.height);

    ImageFileHandle writer (file_name, "wb");
    writer.write_buf (buf);
    writer.close ();
}

template <class SoftImageT>
static void
dump_soft (const SmartPtr<SoftImageT> &image, const char *name)
{
    XCAM_ASSERT (image.ptr ());
    char file_name[256];
    snprintf (file_name, 256, "%s-%dx%d.soft", name, image->get_width(), image->get_height());
    SoftImageFile<SoftImageT> file(file_name, "wb");
    file.write_buf (image);
    file.close ();
}

static
void dump_buf (const SmartPtr<VideoBuffer> buf, const char *name, uint32_t level, uint32_t idx)
{
    char file_name[256];
    XCAM_ASSERT (name);
    snprintf (file_name, 256, "%s-L%d-Idx%d", name, level, idx);
    dump_buf (buf, file_name);
}
#else
static void dump_buf (...) {}
static void dump_soft (...) {}
#endif

SoftBlender::SoftBlender (const char *name)
    : SoftHandler (name)
    , Blender (SOFT_BLENDER_ALIGNMENT_X, SOFT_BLENDER_ALIGNMENT_Y)
{
    _priv_config = new SoftBlenderPriv::BlenderPrivConfig (this, XCAM_SOFT_PYRAMID_DEFAULT_LEVEL);
    XCAM_ASSERT (_priv_config.ptr ());
}

SoftBlender::~SoftBlender ()
{
}

bool
SoftBlender::set_pyr_levels (uint32_t num)
{
    XCAM_ASSERT (num > 0);
    XCAM_FAIL_RETURN (
        ERROR, num > 0, false,
        "blender:%s set_pyr_levels failed, level(%d) must > 0", XCAM_STR (get_name ()), num);

    _priv_config->pyr_levels = num;
    return true;
}

XCamReturn
SoftBlender::blend (
    const SmartPtr<VideoBuffer> &in0,
    const SmartPtr<VideoBuffer> &in1,
    SmartPtr<VideoBuffer> &out_buf)
{
    SmartPtr<BlenderParam> param = new BlenderParam (in0, in1, out_buf);
    XCamReturn ret = execute_buffer (param, true);
    if (xcam_ret_is_ok(ret) && !out_buf.ptr ()) {
        out_buf = param->out_buf;
    }
    return ret;
}

XCamReturn
SoftBlenderPriv::BlenderPrivConfig::init_masks (uint32_t width, uint32_t height)
{
    orig_mask = new UcharImage (
        width, height,
        XCAM_ALIGN_UP (width, SOFT_BLENDER_ALIGNMENT_X));
    XCAM_ASSERT (orig_mask.ptr ());
    XCAM_ASSERT (orig_mask->is_valid ());

    uint32_t left = width / 2;
    uint32_t right = XCAM_ALIGN_UP (width, SOFT_BLENDER_ALIGNMENT_X) - left;
    for (uint32_t h = 0; h < height; ++h) {
        Uchar *ptr = orig_mask->get_buf_ptr (0, h);
        memset (ptr, 255, left);
        memset (ptr + left, 0, right);
    }
    dump_soft (orig_mask, "orig_mask");

    for (uint32_t i = 0; i < pyr_levels; ++i) {
        width = (width + 1) / 2;
        height = (height + 1) / 2;
        pyr_layer[i].coef_mask = new UcharImage (
            width, height,
            XCAM_ALIGN_UP (width, SOFT_BLENDER_ALIGNMENT_X));
        XCAM_ASSERT (pyr_layer[i].coef_mask.ptr ());

        SmartPtr<GaussScaleGray::Args> args = new GaussScaleGray::Args;
        if (i == 0) {
            args->in_luma = orig_mask;
        } else {
            args->in_luma = pyr_layer[i - 1].coef_mask;
        }
        args->out_luma = pyr_layer[i].coef_mask;
        SmartPtr<GaussScaleGray> worker = new GaussScaleGray;
        WorkSize size ((args->out_luma->get_width () + 1) / 2, (args->out_luma->get_height () + 1) / 2);
        worker->set_local_size (size);
        worker->set_global_size (size);
        XCamReturn ret = worker->work (args);
        XCAM_FAIL_RETURN (
            ERROR, xcam_ret_is_ok (ret), ret,
            "blender:(%s) first time scale coeff mask failed. level:%d",
            XCAM_STR (_blender->get_name ()), i);
    }
    dump_soft (pyr_layer[pyr_levels - 1].coef_mask, "orig_last");
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
SoftBlenderPriv::BlenderPrivConfig::start_scaler (
    const SmartPtr<ImageHandler::Parameters> &param,
    const SmartPtr<VideoBuffer> &in_buf,
    const uint32_t level, const SoftBlender::BufIdx idx)
{
    XCAM_ASSERT (level < pyr_levels);
    XCAM_ASSERT (idx < SoftBlender::BufIdxCount);
    SmartPtr<SoftWorker> worker = pyr_layer[level].scale_task[idx];
    XCAM_ASSERT (worker.ptr ());

    XCAM_ASSERT (pyr_layer[level].overlap_pool.ptr ());
    SmartPtr<VideoBuffer> out_buf = pyr_layer[level].overlap_pool->get_buffer ();
    XCAM_FAIL_RETURN (
        ERROR, out_buf.ptr (), XCAM_RETURN_ERROR_MEM,
        "blender:(%s) start_scaler failed, level(%d),idx(%d) get output buffer empty.",
        XCAM_STR (_blender->get_name ()), level, (int)idx);

    SmartPtr<GaussDownScale::Args> args = new GaussDownScale::Args (param, level, idx, in_buf, out_buf);
    args->in_luma = new UcharImage (in_buf, 0);
    args->in_uv = new Uchar2Image (in_buf, 1);
    args->out_luma = new UcharImage (out_buf, 0);
    args->out_uv = new Uchar2Image (out_buf, 1);

    XCAM_ASSERT (out_buf->get_video_info ().width % 2 == 0 && out_buf->get_video_info ().height % 2 == 0);
    uint32_t thread_x = 2, thread_y = 2;
    WorkSize global_size (args->out_uv->get_width (), args->out_uv->get_height ());
    WorkSize local_size (
        xcam_ceil(global_size.value[0], thread_x) / thread_x ,
        xcam_ceil(global_size.value[1], thread_y) / thread_y);

    worker->set_local_size (local_size);
    worker->set_global_size (global_size);

    return worker->work (args);
}

XCamReturn
SoftBlenderPriv::BlenderPrivConfig::start_lap_task (
    const SmartPtr<ImageHandler::Parameters> &param,
    const uint32_t level, const SoftBlender::BufIdx idx,
    const SmartPtr<VideoBuffer> &orig, const SmartPtr<VideoBuffer> &gauss)
{
    XCAM_ASSERT (level < pyr_levels);
    XCAM_ASSERT (idx < SoftBlender::BufIdxCount);

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
        "blender:(%s) start_lap_task failed, level(%d),idx(%d) get output buffer empty.",
        XCAM_STR (_blender->get_name ()), level, (int)idx);

    SmartPtr<LaplaceTask::Args> args = new LaplaceTask::Args (param, level, idx, out_buf);
    args->orig_luma = new UcharImage (orig, 0);
    args->orig_uv = new Uchar2Image (orig, 1);
    args->gauss_luma = new UcharImage (gauss, 0);
    args->gauss_uv = new Uchar2Image (gauss, 1);
    args->out_luma = new UcharImage (out_buf, 0);
    args->out_uv = new Uchar2Image (out_buf, 1);

    uint32_t thread_x = 2, thread_y = 2;
    WorkSize global_size (args->out_uv->get_width () / 4, args->out_uv->get_height () / 2);
    WorkSize local_size (
        xcam_ceil(global_size.value[0], thread_x) / thread_x ,
        xcam_ceil(global_size.value[1], thread_y) / thread_y);


    SmartPtr<SoftWorker> worker = pyr_layer[level].lap_task[idx];
    XCAM_ASSERT (worker.ptr ());
    worker->set_local_size (local_size);
    worker->set_global_size (global_size);

    return worker->work (args);
}

XCamReturn
SoftBlenderPriv::BlenderPrivConfig::start_blend_task (
    const SmartPtr<ImageHandler::Parameters> &param,
    const SmartPtr<VideoBuffer> &buf,
    const SoftBlender::BufIdx idx)
{

    SmartPtr<BlendTask::Args> args;
    uint32_t last_level = pyr_levels - 1;

    {
        SmartLock locker (map_args_mutex);
        MapBlendArgs::iterator i = blend_args.find (param.ptr ());
        if (i == blend_args.end ()) {
            args = new BlendTask::Args (param, pyr_layer[last_level].coef_mask);
            XCAM_ASSERT (args.ptr ());
            blend_args.insert (std::make_pair((void*)param.ptr (), args));
            XCAM_LOG_DEBUG ("soft_blender:%s init blender args", XCAM_STR (_blender->get_name ()));
        } else {
            args = (*i).second;
        }
        args->in_luma[idx] = new UcharImage (buf, 0);
        args->in_uv[idx] = new Uchar2Image (buf, 1);
        XCAM_ASSERT (args->in_luma[idx].ptr () && args->in_uv[idx].ptr ());

        if (!args->in_luma[SoftBlender::Idx0].ptr () || !args->in_luma[SoftBlender::Idx1].ptr ())
            return XCAM_RETURN_BYPASS;

        blend_args.erase (i);
    }

    XCAM_ASSERT (args.ptr ());
    XCAM_ASSERT (args->in_luma[SoftBlender::Idx0]->get_width () == args->in_luma[SoftBlender::Idx1]->get_width ());

    XCAM_ASSERT (pyr_layer[last_level].overlap_pool.ptr ());
    SmartPtr<VideoBuffer> out_buf = pyr_layer[last_level].overlap_pool->get_buffer ();
    XCAM_FAIL_RETURN (
        ERROR, out_buf.ptr (), XCAM_RETURN_ERROR_MEM,
        "blender:(%s) start_blend_task failed, last level blend buffer empty.",
        XCAM_STR (_blender->get_name ()), (int)idx);
    args->out_luma = new UcharImage (out_buf, 0);
    args->out_uv = new Uchar2Image (out_buf, 1);
    args->out_buf = out_buf;

    // process 4x1 uv each loop
    uint32_t thread_x = 2, thread_y = 2;
    WorkSize global_size (args->out_uv->get_width () / 4, args->out_uv->get_height ());
    WorkSize local_size (
        xcam_ceil (global_size.value[0], thread_x) / thread_x,
        xcam_ceil (global_size.value[1], thread_y) / thread_y);

    SmartPtr<SoftWorker> worker = last_level_blend;
    XCAM_ASSERT (worker.ptr ());
    worker->set_local_size (local_size);
    worker->set_global_size (global_size);

    return worker->work (args);
}

XCamReturn
SoftBlenderPriv::BlenderPrivConfig::start_reconstruct_task (
    const SmartPtr<ReconstructTask::Args> &args, const uint32_t level)
{
    XCAM_ASSERT (args.ptr ());
    XCAM_ASSERT (args->lap_luma[SoftBlender::Idx0].ptr () && args->lap_luma[SoftBlender::Idx1].ptr () && args->gauss_luma.ptr ());
    XCAM_ASSERT (args->lap_luma[SoftBlender::Idx0]->get_width () == args->lap_luma[SoftBlender::Idx1]->get_width ());
    SmartPtr<VideoBuffer> out_buf;
    if (level == 0) {
        out_buf = args->get_param ()->out_buf;
        args->mask = orig_mask;
    } else {
        out_buf = pyr_layer[level - 1].overlap_pool->get_buffer ();
        args->mask = pyr_layer[level - 1].coef_mask;
    }
    XCAM_FAIL_RETURN (
        ERROR, out_buf.ptr (), XCAM_RETURN_ERROR_MEM,
        "blender:(%s) start_reconstruct_task failed, out buffer is empty.", XCAM_STR (_blender->get_name ()));
    args->out_luma = new UcharImage (out_buf, 0);
    args->out_uv = new Uchar2Image (out_buf, 1);
    args->out_buf = out_buf;

    uint32_t thread_x = 2, thread_y = 2;
    WorkSize global_size (args->out_uv->get_width (), args->out_uv->get_height ());
    WorkSize local_size (
        xcam_ceil (global_size.value[0], thread_x) / thread_x,
        xcam_ceil (global_size.value[1], thread_y) / thread_y);

    SmartPtr<SoftWorker> worker = pyr_layer[level].recon_task;
    XCAM_ASSERT (worker.ptr ());
    worker->set_local_size (local_size);
    worker->set_global_size (global_size);

    return worker->work (args);
}

XCamReturn
SoftBlenderPriv::BlenderPrivConfig::start_reconstruct_task_by_gauss (
    const SmartPtr<ImageHandler::Parameters> &param,
    const SmartPtr<VideoBuffer> &gauss,
    const uint32_t level)
{
    SmartPtr<ReconstructTask::Args> args;
    {
        SmartLock locker (map_args_mutex);
        MapReconsArgs::iterator i = pyr_layer[level].recons_args.find (param.ptr ());
        if (i == pyr_layer[level].recons_args.end ()) {
            args = new ReconstructTask::Args (param, level);
            XCAM_ASSERT (args.ptr ());
            pyr_layer[level].recons_args.insert (std::make_pair((void*)param.ptr (), args));
            XCAM_LOG_DEBUG ("soft_blender:%s init recons_args level(%d)", XCAM_STR (_blender->get_name ()), level);
        } else {
            args = (*i).second;
        }
        args->gauss_luma = new UcharImage (gauss, 0);
        args->gauss_uv = new Uchar2Image (gauss, 1);
        XCAM_ASSERT (args->gauss_luma.ptr () && args->gauss_uv.ptr ());

        if (!args->lap_luma[SoftBlender::Idx0].ptr () || !args->lap_luma[SoftBlender::Idx1].ptr ())
            return XCAM_RETURN_BYPASS;

        pyr_layer[level].recons_args.erase (i);
    }

    return start_reconstruct_task (args, level);
}

XCamReturn
SoftBlenderPriv::BlenderPrivConfig::start_reconstruct_task_by_lap (
    const SmartPtr<ImageHandler::Parameters> &param,
    const SmartPtr<VideoBuffer> &lap,
    const uint32_t level,
    const SoftBlender::BufIdx idx)
{
    SmartPtr<ReconstructTask::Args> args;
    {
        SmartLock locker (map_args_mutex);
        MapReconsArgs::iterator i = pyr_layer[level].recons_args.find (param.ptr ());
        if (i == pyr_layer[level].recons_args.end ()) {
            args = new ReconstructTask::Args (param, level);
            XCAM_ASSERT (args.ptr ());
            pyr_layer[level].recons_args.insert (std::make_pair((void*)param.ptr (), args));
            XCAM_LOG_DEBUG ("soft_blender:%s init recons_args level(%d)", XCAM_STR (_blender->get_name ()), level);
        } else {
            args = (*i).second;
        }
        args->lap_luma[idx] = new UcharImage (lap, 0);
        args->lap_uv[idx] = new Uchar2Image (lap, 1);
        XCAM_ASSERT (args->lap_luma[idx].ptr () && args->lap_uv[idx].ptr ());

        if (!args->gauss_luma.ptr () || !args->lap_luma[SoftBlender::Idx0].ptr () ||
                !args->lap_luma[SoftBlender::Idx1].ptr ())
            return XCAM_RETURN_BYPASS;

        pyr_layer[level].recons_args.erase (i);
    }

    return start_reconstruct_task (args, level);
}

XCamReturn
SoftBlender::start_work (const SmartPtr<ImageHandler::Parameters> &base)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    SmartPtr<BlenderParam> param = base.dynamic_cast_ptr<BlenderParam> ();

    XCAM_FAIL_RETURN (
        ERROR, param.ptr () && param->in1_buf.ptr () && param->in_buf.ptr (), XCAM_RETURN_ERROR_PARAM,
        "blender:%s start_work failed, params(in0/in1) are not fully set or type not correct",
        XCAM_STR (get_name ()));

    if (!param->out_buf.ptr ()) {
        param->out_buf = get_free_buf ();
    }

    XCAM_FAIL_RETURN (
        ERROR, param->out_buf.ptr (), XCAM_RETURN_ERROR_PARAM,
        "blender:%s start_work failed, params output buffer was not set or failed in allocation.",
        XCAM_STR (get_name ()));

    //start gauss scale level0: idx0
    ret = _priv_config->start_scaler (param, param->in_buf, 0, Idx0);
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret,
        "blender:%s start_work failed on idx0", XCAM_STR (get_name ()));

    //start gauss scale level0: idx1
    ret = _priv_config->start_scaler (param, param->in1_buf, 0, Idx1);
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret,
        "blender:%s start_work failed on idx1", XCAM_STR (get_name ()));

    param->in_buf.release ();
    param->in1_buf.release ();

    return ret;
};

XCamReturn
SoftBlender::configure_resource (const SmartPtr<Parameters> &param)
{
    XCAM_ASSERT (_priv_config->pyr_levels <= XCAM_SOFT_PYRAMID_MAX_LEVEL);
    const VideoBufferInfo &in0_info = param->in_buf->get_video_info ();
    XCAM_FAIL_RETURN (
        ERROR, in0_info.format == V4L2_PIX_FMT_NV12, XCAM_RETURN_ERROR_PARAM,
        "blender:%s only support format(NV12) but input format is %s",
        XCAM_STR(get_name ()), xcam_fourcc_to_string (in0_info.format));

    VideoBufferInfo out_info;
    uint32_t out_width(0), out_height(0);
    get_output_size (out_width, out_height);
    XCAM_FAIL_RETURN (
        ERROR, out_width && out_height, XCAM_RETURN_ERROR_PARAM,
        "blender:%s output size was not set", XCAM_STR(get_name ()));

    out_info.init (
        in0_info.format, out_width, out_height,
        XCAM_ALIGN_UP (out_width, SOFT_BLENDER_ALIGNMENT_X), XCAM_ALIGN_UP (out_height, SOFT_BLENDER_ALIGNMENT_Y));
    set_out_video_info (out_info);

    VideoBufferInfo overlap_info;
    Rect merge_size = get_merge_window ();
    //overlap_info.init (in0_info.format, merge_size.width, merge_size.height);
    XCAM_ASSERT (merge_size.width % SOFT_BLENDER_ALIGNMENT_X == 0);

    overlap_info.init (in0_info.format, merge_size.width, merge_size.height);
    _priv_config->first_lap_pool = new SoftVideoBufAllocator (overlap_info);
    XCAM_FAIL_RETURN (
        ERROR, _priv_config->first_lap_pool->reserve (LAP_POOL_SIZE), XCAM_RETURN_ERROR_MEM,
        "blender:%s reserve lap buffer pool(w:%d,h:%d) failed",
        XCAM_STR(get_name ()), overlap_info.width, overlap_info.height);

    SmartPtr<Worker::Callback> gauss_scale_cb = new CbGaussDownScale (this);
    SmartPtr<Worker::Callback> lap_cb = new CbLapTask (this);
    SmartPtr<Worker::Callback> reconst_cb = new CbReconstructTask (this);
    XCAM_ASSERT (gauss_scale_cb.ptr () && lap_cb.ptr () && reconst_cb.ptr ());

    XCamReturn ret = _priv_config->init_masks (merge_size.width, merge_size.height);
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret,
        "blender:%s init masks failed", XCAM_STR (get_name ()));

    for (uint32_t i = 0; i < _priv_config->pyr_levels; ++i) {
        merge_size.width = XCAM_ALIGN_UP ((merge_size.width + 1) / 2, SOFT_BLENDER_ALIGNMENT_X);
        merge_size.height = XCAM_ALIGN_UP ((merge_size.height + 1) / 2, SOFT_BLENDER_ALIGNMENT_Y);
        overlap_info.init (in0_info.format, merge_size.width, merge_size.height);

        _priv_config->pyr_layer[i].overlap_pool = new SoftVideoBufAllocator (overlap_info);
        XCAM_ASSERT (_priv_config->pyr_layer[i].overlap_pool.ptr ());
        XCAM_FAIL_RETURN (
            ERROR, _priv_config->pyr_layer[i].overlap_pool->reserve (OVERLAP_POOL_SIZE), XCAM_RETURN_ERROR_MEM,
            "blender:%s reserve buffer pool(w:%d,h:%d) failed",
            XCAM_STR(get_name ()), overlap_info.width, overlap_info.height);

        _priv_config->pyr_layer[i].scale_task[SoftBlender::Idx0] = new GaussDownScale (gauss_scale_cb);
        XCAM_ASSERT (_priv_config->pyr_layer[i].scale_task[SoftBlender::Idx0].ptr ());
        _priv_config->pyr_layer[i].scale_task[SoftBlender::Idx1] = new GaussDownScale (gauss_scale_cb);
        XCAM_ASSERT (_priv_config->pyr_layer[i].scale_task[SoftBlender::Idx1].ptr ());
        _priv_config->pyr_layer[i].lap_task[SoftBlender::Idx0] = new LaplaceTask (lap_cb);
        XCAM_ASSERT (_priv_config->pyr_layer[i].lap_task[SoftBlender::Idx0].ptr ());
        _priv_config->pyr_layer[i].lap_task[SoftBlender::Idx1] = new LaplaceTask (lap_cb);
        XCAM_ASSERT (_priv_config->pyr_layer[i].lap_task[SoftBlender::Idx1].ptr ());
        _priv_config->pyr_layer[i].recon_task = new ReconstructTask (reconst_cb);
        XCAM_ASSERT (_priv_config->pyr_layer[i].recon_task.ptr ());
    }

    _priv_config->last_level_blend = new BlendTask (new CbBlendTask (this));
    XCAM_ASSERT (_priv_config->last_level_blend.ptr ());

    return XCAM_RETURN_NO_ERROR;
}

void
SoftBlender::gauss_scale_done (
    const SmartPtr<Worker> &worker, const SmartPtr<Worker::Arguments> &base, const XCamReturn error)
{
    XCAM_UNUSED (worker);

    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    SmartPtr<GaussDownScale::Args> args = base.dynamic_cast_ptr<GaussDownScale::Args> ();
    XCAM_ASSERT (args.ptr ());
    const SmartPtr<ImageHandler::Parameters> param = args->get_param ();
    uint32_t level = args->level;
    BufIdx idx = args->idx;
    uint32_t next_level = level + 1;

    XCAM_ASSERT (param.ptr ());
    XCAM_ASSERT (level < _priv_config->pyr_levels);

    if (!check_work_continue (param, error))
        return;

    dump_buf (args->out_buf, "gauss-scale", level, idx);

    ret = _priv_config->start_lap_task (param, level, idx, args->in_buf, args->out_buf);
    if (!xcam_ret_is_ok (ret)) {
        work_broken (param, ret);
    }

    if (next_level == _priv_config->pyr_levels) { // last level
        ret = _priv_config->start_blend_task (param, args->out_buf, idx);
    } else {
        ret = _priv_config->start_scaler (param, args->out_buf, next_level, idx);
    }

    if (!xcam_ret_is_ok (ret)) {
        work_broken (param, ret);
    }
}

void
SoftBlender::lap_done (
    const SmartPtr<Worker> &worker, const SmartPtr<Worker::Arguments> &base, const XCamReturn error)
{
    XCAM_UNUSED (worker);

    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    SmartPtr<LaplaceTask::Args> args = base.dynamic_cast_ptr<LaplaceTask::Args> ();
    XCAM_ASSERT (args.ptr ());
    const SmartPtr<ImageHandler::Parameters> param = args->get_param ();
    XCAM_ASSERT (param.ptr ());
    uint32_t level = args->level;
    BufIdx idx = args->idx;
    XCAM_ASSERT (level < _priv_config->pyr_levels);

    if (!check_work_continue (param, error))
        return;

    dump_buf (args->out_buf, "lap", level, idx);

    ret = _priv_config->start_reconstruct_task_by_lap (param, args->out_buf, level, idx);

    if (!xcam_ret_is_ok (ret)) {
        work_broken (param, ret);
    }
}

void
SoftBlender::blend_task_done (
    const SmartPtr<Worker> &worker, const SmartPtr<Worker::Arguments> &base, const XCamReturn error)
{
    XCAM_UNUSED (worker);

    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    SmartPtr<BlendTask::Args> args = base.dynamic_cast_ptr<BlendTask::Args> ();
    XCAM_ASSERT (args.ptr ());
    const SmartPtr<ImageHandler::Parameters> param = args->get_param ();
    XCAM_ASSERT (param.ptr ());

    if (!check_work_continue (param, error))
        return;

    dump_buf (args->out_buf, "blend-last");
    ret = _priv_config->start_reconstruct_task_by_gauss (param, args->out_buf, _priv_config->pyr_levels - 1);

    if (!xcam_ret_is_ok (ret)) {
        work_broken (param, ret);
    }
}

void
SoftBlender::reconstruct_done (
    const SmartPtr<Worker> &worker, const SmartPtr<Worker::Arguments> &base, const XCamReturn error)
{
    XCAM_UNUSED (worker);

    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    SmartPtr<ReconstructTask::Args> args = base.dynamic_cast_ptr<ReconstructTask::Args> ();
    XCAM_ASSERT (args.ptr ());
    const SmartPtr<ImageHandler::Parameters> param = args->get_param ();
    XCAM_ASSERT (param.ptr ());
    uint32_t level = args->level;
    XCAM_ASSERT (level < _priv_config->pyr_levels);

    if (!check_work_continue (param, error))
        return;

    dump_buf (args->out_buf, "reconstruct", level, 0);

    if (level == 0) {
        work_well_done (param, error);
        return;
    }

    ret = _priv_config->start_reconstruct_task_by_gauss (param, args->out_buf, level - 1);
    if (!xcam_ret_is_ok (ret)) {
        work_broken (param, ret);
    }
}

SmartPtr<SoftHandler>
create_soft_blender ()
{
    SmartPtr<SoftBlender> blender = new SoftBlender();
    XCAM_ASSERT (blender.ptr ());
    return blender;
}

SmartPtr<Blender>
Blender::create_soft_blender ()
{
    SmartPtr<SoftHandler> handler = XCam::create_soft_blender ();
    return handler.dynamic_cast_ptr<Blender> ();
}

}
