/*
 * fake_poll_thread.cpp - poll thread for raw image
 *
 *  Copyright (c) 2014-2015 Intel Corporation
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
 * Author: Jia Meng <jia.meng@intel.com>
 */

#include "fake_poll_thread.h"
#if HAVE_LIBDRM
#include "drm_bo_buffer.h"
#endif

#define DEFAULT_FPT_BUF_COUNT 4

namespace XCam {

FakePollThread::FakePollThread (const char *raw_path)
    : _raw_path (NULL)
    , _raw (NULL)
{
    XCAM_ASSERT (raw_path);

    if (raw_path)
        _raw_path = strndup (raw_path, XCAM_MAX_STR_SIZE);
}

FakePollThread::~FakePollThread ()
{
    if (_raw_path)
        xcam_free (_raw_path);

    if (_raw)
        fclose (_raw);
}

XCamReturn
FakePollThread::start()
{
    XCAM_FAIL_RETURN(
        ERROR,
        _raw_path,
        XCAM_RETURN_ERROR_FILE,
        "FakePollThread failed due to raw path NULL");

    _raw = fopen (_raw_path, "rb");
    XCAM_FAIL_RETURN(
        ERROR,
        _raw,
        XCAM_RETURN_ERROR_FILE,
        "FakePollThread failed to open file:%s", XCAM_STR (_raw_path));

    return PollThread::start ();
}

XCamReturn
FakePollThread::stop ()
{
    if (_buf_pool.ptr ())
        _buf_pool->stop ();

    return PollThread::stop ();;
}

XCamReturn
FakePollThread::read_buf (SmartPtr<VideoBuffer> &buf)
{
    uint8_t *dst = buf->map ();
    const VideoBufferInfo info = buf->get_video_info ();
    VideoBufferPlanarInfo planar;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    for (uint32_t index = 0; index < info.components; index++) {
        info.get_planar_info(planar, index);
        uint32_t line_bytes = planar.width * planar.pixel_bytes;

        for (uint32_t i = 0; i < planar.height; i++) {
            if (fread (dst + info.offsets [index] + i * info.strides [index], 1, line_bytes, _raw) < line_bytes) {
                if (feof (_raw)) {
                    fseek (_raw, 0, SEEK_SET);
                    ret = XCAM_RETURN_BYPASS;
                } else {
                    XCAM_LOG_ERROR ("poll_buffer_loop failed to read file");
                    ret = XCAM_RETURN_ERROR_FILE;
                }
                goto done;
            }
        }
    }

done:
    buf->unmap ();
    return ret;
}

XCamReturn
FakePollThread::poll_buffer_loop ()
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    if (!_buf_pool.ptr () && init_buffer_pool () != XCAM_RETURN_NO_ERROR)
        return XCAM_RETURN_ERROR_MEM;

    SmartPtr<VideoBuffer> buf = _buf_pool->get_buffer (_buf_pool);
    if (!buf.ptr ()) {
        XCAM_LOG_WARNING ("FakePollThread get buffer failed");
        return XCAM_RETURN_ERROR_MEM;
    }

    ret = read_buf (buf);
    if (ret == XCAM_RETURN_BYPASS) {
        ret = read_buf (buf);
    }

    SmartPtr<VideoBuffer> video_buf = buf;
    if (ret == XCAM_RETURN_NO_ERROR && _poll_callback)
        return _poll_callback->poll_buffer_ready (video_buf);

    return ret;
}

XCamReturn
FakePollThread::init_buffer_pool ()
{
    struct v4l2_format format;
    if (!_capture_dev.ptr () ||
            _capture_dev->get_format (format) != XCAM_RETURN_NO_ERROR) {
        XCAM_LOG_ERROR ("Can't init buffer pool without format");
        return XCAM_RETURN_ERROR_PARAM;
    }
    VideoBufferInfo info;
    info.init(format.fmt.pix.pixelformat,
              format.fmt.pix.width,
              format.fmt.pix.height, 0, 0, 0);
#if HAVE_LIBDRM
    SmartPtr<DrmDisplay> drm_disp = DrmDisplay::instance ();
    SmartPtr<BufferPool> pool = new DrmBoBufferPool (drm_disp);
    XCAM_ASSERT (pool.ptr ());
    _buf_pool = pool;

    if (_buf_pool->set_video_info (info) && _buf_pool->reserve (DEFAULT_FPT_BUF_COUNT))
        return XCAM_RETURN_NO_ERROR;
#endif

    return XCAM_RETURN_ERROR_MEM;
}

};
