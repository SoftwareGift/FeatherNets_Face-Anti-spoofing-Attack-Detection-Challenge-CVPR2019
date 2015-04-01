/*
 * stub.cpp - stub utilities that implemented in CPP
 *
 *  Copyright (c) 2015 Intel Corporation
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
 * Author: John Ye <john.ye@intel.com>
 */

#include "stub.h"
#include "bufmap.h"
#include "v4l2dev.h"

#include <stdio.h>

using namespace XCam;

int libxcam_dequeue_buffer (SmartPtr<V4l2BufferProxy> &buf)
{
    SmartPtr<MainDeviceManager> device_manager = DeviceManagerInstance::device_manager_instance();
    int ret;
    SmartPtr<VideoBuffer> video_buffer;

    pthread_mutex_lock (&device_manager->bufs_mutex);
    if (device_manager->bufs.size() == 0) {
        pthread_cond_wait (&device_manager->bufs_cond, &device_manager->bufs_mutex);
    }
    video_buffer = device_manager->bufs.front();
    device_manager->bufs.pop();
    pthread_mutex_unlock (&device_manager->bufs_mutex);
    buf = video_buffer.dynamic_cast_ptr<V4l2BufferProxy>();

    pthread_mutex_lock (&device_manager->release_mutex);
    device_manager->release_bufs.push (video_buffer);
    pthread_mutex_unlock (&device_manager->release_mutex);
    return (int) XCAM_RETURN_NO_ERROR;
}

GstFlowReturn
xcam_bufferpool_acquire_buffer (GstBufferPool *bpool, GstBuffer **buffer, GstBufferPoolAcquireParams *params)
{
    GstBuffer *gbuf = NULL;
    Gstxcambufferpool *pool = GST_XCAMBUFFERPOOL_CAST (bpool);
    Gstxcamsrc *xcamsrc = pool->src;

    SmartPtr<V4l2BufferProxy> buf;
    libxcam_dequeue_buffer (buf);

    SmartPtr<BufMap> bufmap = BufMap::instance();
    gbuf = bufmap->gbuf(buf);

    if (!gbuf) {
        gbuf = gst_buffer_new();
        GST_BUFFER (gbuf)->pool = (GstBufferPool*) pool;

        gst_buffer_append_memory (gbuf,
                                  gst_dmabuf_allocator_alloc (pool->allocator, buf->get_v4l2_dma_fd(), buf->get_v4l2_buf_length()));
        bufmap->setmap(gbuf, buf);
    }

    GST_BUFFER_TIMESTAMP (gbuf) = buf->get_timestamp();
    *buffer = gbuf;

    return GST_FLOW_OK;
}

void
xcambufferpool_release_buffer (GstBufferPool *bpool, GstBuffer *gbuf)
{
    SmartPtr<MainDeviceManager> device_manager = DeviceManagerInstance::device_manager_instance();
    pthread_mutex_lock (&device_manager->release_mutex);
    device_manager->release_bufs.pop();
    pthread_mutex_unlock (&device_manager->release_mutex);
}

gboolean gst_xcamsrc_set_white_balance_mode (GstXCam3A *xcam3a, XCamAwbMode mode)
{
    SmartPtr<MainDeviceManager> device_manager = DeviceManagerInstance::device_manager_instance();
    SmartPtr<X3aAnalyzer> analyzer = device_manager->get_x3a_analyzer ();
    return analyzer->set_awb_mode (mode);
}

gboolean gst_xcamsrc_set_awb_speed (GstXCam3A *xcam3a, double speed)
{
    SmartPtr<MainDeviceManager> device_manager = DeviceManagerInstance::device_manager_instance();
    SmartPtr<X3aAnalyzer> analyzer = device_manager->get_x3a_analyzer ();
    return analyzer->set_awb_speed (speed);
}

gboolean gst_xcamsrc_set_wb_color_temperature_range (GstXCam3A *xcam3a, guint cct_min, guint cct_max)
{
    SmartPtr<MainDeviceManager> device_manager = DeviceManagerInstance::device_manager_instance();
    SmartPtr<X3aAnalyzer> analyzer = device_manager->get_x3a_analyzer ();
    return analyzer->set_awb_color_temperature_range (cct_min, cct_max);
}
gboolean gst_xcamsrc_set_manual_wb_gain (GstXCam3A *xcam3a, double gr, double r, double b, double gb)
{
    SmartPtr<MainDeviceManager> device_manager = DeviceManagerInstance::device_manager_instance();
    SmartPtr<X3aAnalyzer> analyzer = device_manager->get_x3a_analyzer ();
    return analyzer->set_awb_manual_gain (gr, r, b, gb);
}
gboolean gst_xcamsrc_set_exposure_mode (GstXCam3A *xcam3a, XCamAeMode mode)
{
    SmartPtr<MainDeviceManager> device_manager = DeviceManagerInstance::device_manager_instance();
    SmartPtr<X3aAnalyzer> analyzer = device_manager->get_x3a_analyzer ();
    return analyzer->set_ae_mode (mode);
}

gboolean gst_xcamsrc_set_ae_metering_mode (GstXCam3A *xcam3a, XCamAeMeteringMode mode)
{
    SmartPtr<MainDeviceManager> device_manager = DeviceManagerInstance::device_manager_instance();
    SmartPtr<X3aAnalyzer> analyzer = device_manager->get_x3a_analyzer ();
    return analyzer->set_ae_metering_mode (mode);
}
gboolean gst_xcamsrc_set_exposure_window (GstXCam3A *xcam3a, XCam3AWindow *window)
{
    SmartPtr<MainDeviceManager> device_manager = DeviceManagerInstance::device_manager_instance();
    SmartPtr<X3aAnalyzer> analyzer = device_manager->get_x3a_analyzer ();
    return analyzer->set_ae_window (window);
}
gboolean gst_xcamsrc_set_exposure_value_offset (GstXCam3A *xcam3a, double ev_offset)
{
    SmartPtr<MainDeviceManager> device_manager = DeviceManagerInstance::device_manager_instance();
    SmartPtr<X3aAnalyzer> analyzer = device_manager->get_x3a_analyzer ();
    return analyzer->set_ae_ev_shift (ev_offset);
}
gboolean gst_xcamsrc_set_ae_speed (GstXCam3A *xcam3a, double speed)
{
    SmartPtr<MainDeviceManager> device_manager = DeviceManagerInstance::device_manager_instance();
    SmartPtr<X3aAnalyzer> analyzer = device_manager->get_x3a_analyzer ();
    return analyzer->set_ae_speed (speed);
}
gboolean gst_xcamsrc_set_exposure_flicker_mode (GstXCam3A *xcam3a, XCamFlickerMode flicker)
{
    SmartPtr<MainDeviceManager> device_manager = DeviceManagerInstance::device_manager_instance();
    SmartPtr<X3aAnalyzer> analyzer = device_manager->get_x3a_analyzer ();
    return analyzer->set_ae_flicker_mode (flicker);
}
XCamFlickerMode gst_xcamsrc_get_exposure_flicker_mode (GstXCam3A *xcam3a)
{
    SmartPtr<MainDeviceManager> device_manager = DeviceManagerInstance::device_manager_instance();
    SmartPtr<X3aAnalyzer> analyzer = device_manager->get_x3a_analyzer ();
    return analyzer->get_ae_flicker_mode ();
}
gint64 gst_xcamsrc_get_current_exposure_time (GstXCam3A *xcam3a)
{
    SmartPtr<MainDeviceManager> device_manager = DeviceManagerInstance::device_manager_instance();
    SmartPtr<X3aAnalyzer> analyzer = device_manager->get_x3a_analyzer ();
    return analyzer->get_ae_current_exposure_time ();
}
double gst_xcamsrc_get_current_analog_gain (GstXCam3A *xcam3a)
{
    SmartPtr<MainDeviceManager> device_manager = DeviceManagerInstance::device_manager_instance();
    SmartPtr<X3aAnalyzer> analyzer = device_manager->get_x3a_analyzer ();
    return analyzer->get_ae_current_analog_gain ();
}
gboolean gst_xcamsrc_set_manual_exposure_time (GstXCam3A *xcam3a, gint64 time_in_us)
{
    SmartPtr<MainDeviceManager> device_manager = DeviceManagerInstance::device_manager_instance();
    SmartPtr<X3aAnalyzer> analyzer = device_manager->get_x3a_analyzer ();
    return analyzer->set_ae_manual_exposure_time (time_in_us);
}
gboolean gst_xcamsrc_set_manual_analog_gain (GstXCam3A *xcam3a, double gain)
{
    SmartPtr<MainDeviceManager> device_manager = DeviceManagerInstance::device_manager_instance();
    SmartPtr<X3aAnalyzer> analyzer = device_manager->get_x3a_analyzer ();
    return analyzer->set_ae_manual_analog_gain (gain);
}
gboolean gst_xcamsrc_set_aperture (GstXCam3A *xcam3a, double fn)
{
    SmartPtr<MainDeviceManager> device_manager = DeviceManagerInstance::device_manager_instance();
    SmartPtr<X3aAnalyzer> analyzer = device_manager->get_x3a_analyzer ();
    return analyzer->set_ae_aperture (fn);
}
gboolean gst_xcamsrc_set_max_analog_gain (GstXCam3A *xcam3a, double max_gain)
{
    SmartPtr<MainDeviceManager> device_manager = DeviceManagerInstance::device_manager_instance();
    SmartPtr<X3aAnalyzer> analyzer = device_manager->get_x3a_analyzer ();
    return analyzer->set_ae_max_analog_gain (max_gain);
}
double gst_xcamsrc_get_max_analog_gain (GstXCam3A *xcam3a)
{
    SmartPtr<MainDeviceManager> device_manager = DeviceManagerInstance::device_manager_instance();
    SmartPtr<X3aAnalyzer> analyzer = device_manager->get_x3a_analyzer ();
    return analyzer->get_ae_max_analog_gain ();
}
gboolean gst_xcamsrc_set_exposure_time_range (GstXCam3A *xcam3a, gint64 min_time_in_us, gint64 max_time_in_us)
{
    SmartPtr<MainDeviceManager> device_manager = DeviceManagerInstance::device_manager_instance();
    SmartPtr<X3aAnalyzer> analyzer = device_manager->get_x3a_analyzer ();
    return analyzer->set_ae_exposure_time_range (min_time_in_us, max_time_in_us);
}
gboolean gst_xcamsrc_get_exposure_time_range (GstXCam3A *xcam3a, gint64 *min_time_in_us, gint64 *max_time_in_us)
{
    SmartPtr<MainDeviceManager> device_manager = DeviceManagerInstance::device_manager_instance();
    SmartPtr<X3aAnalyzer> analyzer = device_manager->get_x3a_analyzer ();
    return analyzer->get_ae_exposure_time_range (min_time_in_us, max_time_in_us);
}
gboolean gst_xcamsrc_set_noise_reduction_level (GstXCam3A *xcam3a, guint8 level)
{
    SmartPtr<MainDeviceManager> device_manager = DeviceManagerInstance::device_manager_instance();
    SmartPtr<X3aAnalyzer> analyzer = device_manager->get_x3a_analyzer ();
    return analyzer->set_noise_reduction_level (level);
}
gboolean gst_xcamsrc_set_temporal_noise_reduction_level (GstXCam3A *xcam3a, guint8 level)
{
    SmartPtr<MainDeviceManager> device_manager = DeviceManagerInstance::device_manager_instance();
    SmartPtr<X3aAnalyzer> analyzer = device_manager->get_x3a_analyzer ();
    return analyzer->set_temporal_noise_reduction_level (level);
}
gboolean gst_xcamsrc_set_gamma_table (GstXCam3A *xcam3a, double *r_table, double *g_table, double *b_table)
{
    SmartPtr<MainDeviceManager> device_manager = DeviceManagerInstance::device_manager_instance();
    SmartPtr<X3aAnalyzer> analyzer = device_manager->get_x3a_analyzer ();
    return analyzer->set_gamma_table (r_table, g_table, b_table);
}
gboolean gst_xcamsrc_set_gbce (GstXCam3A *xcam3a, gboolean enable)
{
    SmartPtr<MainDeviceManager> device_manager = DeviceManagerInstance::device_manager_instance();
    SmartPtr<X3aAnalyzer> analyzer = device_manager->get_x3a_analyzer ();
    return analyzer->set_gbce (enable);
}
gboolean gst_xcamsrc_set_manual_brightness (GstXCam3A *xcam3a, guint8 value)
{
    SmartPtr<MainDeviceManager> device_manager = DeviceManagerInstance::device_manager_instance();
    SmartPtr<X3aAnalyzer> analyzer = device_manager->get_x3a_analyzer ();
    return analyzer->set_manual_brightness (value);
}
gboolean gst_xcamsrc_set_manual_contrast (GstXCam3A *xcam3a, guint8 value)
{
    SmartPtr<MainDeviceManager> device_manager = DeviceManagerInstance::device_manager_instance();
    SmartPtr<X3aAnalyzer> analyzer = device_manager->get_x3a_analyzer ();
    return analyzer->set_manual_contrast (value);
}
gboolean gst_xcamsrc_set_manual_hue (GstXCam3A *xcam3a, guint8 value)
{
    SmartPtr<MainDeviceManager> device_manager = DeviceManagerInstance::device_manager_instance();
    SmartPtr<X3aAnalyzer> analyzer = device_manager->get_x3a_analyzer ();
    return analyzer->set_manual_hue (value);
}
gboolean gst_xcamsrc_set_manual_saturation (GstXCam3A *xcam3a, guint8 value)
{
    SmartPtr<MainDeviceManager> device_manager = DeviceManagerInstance::device_manager_instance();
    SmartPtr<X3aAnalyzer> analyzer = device_manager->get_x3a_analyzer ();
    return analyzer->set_manual_saturation (value);
}
gboolean gst_xcamsrc_set_manual_sharpness (GstXCam3A *xcam3a, guint8 value)
{
    SmartPtr<MainDeviceManager> device_manager = DeviceManagerInstance::device_manager_instance();
    SmartPtr<X3aAnalyzer> analyzer = device_manager->get_x3a_analyzer ();
    return analyzer->set_manual_sharpness (value);
}
gboolean gst_xcamsrc_set_dvs (GstXCam3A *xcam3a, gboolean enable)
{
    SmartPtr<MainDeviceManager> device_manager = DeviceManagerInstance::device_manager_instance();
    SmartPtr<X3aAnalyzer> analyzer = device_manager->get_x3a_analyzer ();
    return analyzer->set_dvs (enable);
}
gboolean gst_xcamsrc_set_night_mode (GstXCam3A *xcam3a, gboolean enable)
{
    SmartPtr<MainDeviceManager> device_manager = DeviceManagerInstance::device_manager_instance();
    SmartPtr<X3aAnalyzer> analyzer = device_manager->get_x3a_analyzer ();
    return analyzer->set_night_mode (enable);
}

