/*
 * gstxcaminterface.c - Gstreamer XCam 3A interface
 *
 * Copyright (C) 2014 Intel Corporation
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library. If not, see <http://www.gnu.org/licenses/>.
 *
 * Author: Wind Yuan <feng.yuan@intel.com>
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "gstxcaminterface.h"
#include <string.h>

static void gst_xcam_3a_iface_init (GstXCam3AInterface *iface);

GType
gst_xcam_3a_interface_get_type (void)
{
    static GType gst_xcam_3a_interface_type = 0;

    if (!gst_xcam_3a_interface_type) {
        static const GTypeInfo gst_xcam_3a_interface_info = {
            sizeof (GstXCam3AInterface),
            (GBaseInitFunc) gst_xcam_3a_iface_init,
            NULL,
            NULL,
            NULL,
            NULL,
            0,
            0,
            NULL,
        };

        gst_xcam_3a_interface_type = g_type_register_static (G_TYPE_INTERFACE,
                                     "GsXCam3AInterface", &gst_xcam_3a_interface_info, 0);
    }
    return gst_xcam_3a_interface_type;
}

static void
gst_xcam_3a_iface_init (GstXCam3AInterface * iface)
{
    /* default virtual functions */
    iface->set_white_balance_mode = NULL;
    iface->set_awb_speed = NULL;
    iface->set_wb_color_temperature_range = NULL;
    iface->set_manual_wb_gain = NULL;
    iface->set_exposure_mode = NULL;
    iface->set_ae_metering_mode = NULL;
    iface->set_exposure_window = NULL;
    iface->set_exposure_value_offset = NULL;
    iface->set_ae_speed = NULL;
    iface->set_exposure_flicker_mode = NULL;
    iface->get_exposure_flicker_mode = NULL;
    iface->get_current_exposure_time = NULL;
    iface->get_current_analog_gain = NULL;
    iface->set_manual_exposure_time = NULL;
    iface->set_manual_analog_gain = NULL;
    iface->set_aperture = NULL;
    iface->set_max_analog_gain = NULL;
    iface->get_max_analog_gain = NULL;
    iface->set_exposure_time_range = NULL;
    iface->get_exposure_time_range = NULL;
    iface->set_dvs = NULL;
    iface->set_noise_reduction_level = NULL;
    iface->set_temporal_noise_reduction_level = NULL;
    iface->set_gamma_table = NULL;
    iface->set_gbce = NULL;
    iface->set_manual_brightness = NULL;
    iface->set_manual_contrast = NULL;
    iface->set_manual_hue = NULL;
    iface->set_manual_saturation = NULL;
    iface->set_manual_sharpness = NULL;
    iface->set_night_mode = NULL;
    iface->set_3a_mode = NULL;
}
