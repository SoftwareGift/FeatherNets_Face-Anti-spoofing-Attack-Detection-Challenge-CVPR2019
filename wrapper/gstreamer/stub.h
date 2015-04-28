/*
 * stub.h - stub utilities that implemented in CPP
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

#ifndef __STUB_H__
#define __STUB_H__

#include <xcam_defs.h>
#include <linux/videodev2.h>

#include <gst/gst.h>
#include <gst/allocators/allocators.h>
#include <gst/video/gstvideopool.h>
#include "gstxcaminterface.h"
#include "gstxcambufferpool.h"
extern "C" {
#include <drm.h>
#include <drm_mode.h>
#include <intel_bufmgr.h>
}

XCAM_BEGIN_DECLARE

enum v4l2_memory;
enum v4l2_field;
struct v4l2_format;
struct v4l2_buffer;

GstFlowReturn xcam_bufferpool_acquire_buffer (GstBufferPool *bpool, GstBuffer **buffer, GstBufferPoolAcquireParams *params);
void xcambufferpool_release_buffer (GstBufferPool *bpool, GstBuffer *buffer);

gboolean gst_xcamsrc_set_white_balance_mode (GstXCam3A *xcam3a, XCamAwbMode mode);
gboolean gst_xcamsrc_set_awb_speed (GstXCam3A *xcam3a, double speed);
gboolean gst_xcamsrc_set_wb_color_temperature_range (GstXCam3A *xcam3a, guint cct_min, guint cct_max);
gboolean gst_xcamsrc_set_manual_wb_gain (GstXCam3A *xcam3a, double gr, double r, double b, double gb);
gboolean gst_xcamsrc_set_exposure_mode (GstXCam3A *xcam3a, XCamAeMode mode);
gboolean gst_xcamsrc_set_ae_metering_mode (GstXCam3A *xcam3a, XCamAeMeteringMode mode);
gboolean gst_xcamsrc_set_exposure_window (GstXCam3A *xcam3a, XCam3AWindow *window, guint8 count = 1);
gboolean gst_xcamsrc_set_exposure_value_offset (GstXCam3A *xcam3a, double ev_offset);
gboolean gst_xcamsrc_set_ae_speed (GstXCam3A *xcam3a, double speed);
gboolean gst_xcamsrc_set_exposure_flicker_mode (GstXCam3A *xcam3a, XCamFlickerMode flicker);
XCamFlickerMode gst_xcamsrc_get_exposure_flicker_mode (GstXCam3A *xcam3a);
gint64 gst_xcamsrc_get_current_exposure_time (GstXCam3A *xcam3a);
double gst_xcamsrc_get_current_analog_gain (GstXCam3A *xcam3a);
gboolean gst_xcamsrc_set_manual_exposure_time (GstXCam3A *xcam3a, gint64 time_in_us);
gboolean gst_xcamsrc_set_manual_analog_gain (GstXCam3A *xcam3a, double gain);
gboolean gst_xcamsrc_set_aperture (GstXCam3A *xcam3a, double fn);
gboolean gst_xcamsrc_set_max_analog_gain (GstXCam3A *xcam3a, double max_gain);
double gst_xcamsrc_get_max_analog_gain (GstXCam3A *xcam3a);
gboolean gst_xcamsrc_set_exposure_time_range (GstXCam3A *xcam3a, gint64 min_time_in_us, gint64 max_time_in_us);
gboolean gst_xcamsrc_get_exposure_time_range (GstXCam3A *xcam3a, gint64 *min_time_in_us, gint64 *max_time_in_us);
gboolean gst_xcamsrc_set_noise_reduction_level (GstXCam3A *xcam3a, guint8 level);
gboolean gst_xcamsrc_set_temporal_noise_reduction_level (GstXCam3A *xcam3a, guint8 level);
gboolean gst_xcamsrc_set_gamma_table (GstXCam3A *xcam3a, double *r_table, double *g_table, double *b_table);
gboolean gst_xcamsrc_set_gbce (GstXCam3A *xcam3a, gboolean enable);
gboolean gst_xcamsrc_set_manual_brightness (GstXCam3A *xcam3a, guint8 value);
gboolean gst_xcamsrc_set_manual_contrast (GstXCam3A *xcam3a, guint8 value);
gboolean gst_xcamsrc_set_manual_hue (GstXCam3A *xcam3a, guint8 value);
gboolean gst_xcamsrc_set_manual_saturation (GstXCam3A *xcam3a, guint8 value);
gboolean gst_xcamsrc_set_manual_sharpness (GstXCam3A *xcam3a, guint8 value);
gboolean gst_xcamsrc_set_dvs (GstXCam3A *xcam3a, gboolean enable);
gboolean gst_xcamsrc_set_night_mode (GstXCam3A *xcam3a, gboolean enable);

XCAM_END_DECLARE

#endif  //__STUB_H__
