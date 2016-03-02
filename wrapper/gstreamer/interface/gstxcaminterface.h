/*
 * gstxcaminterface.h - gst xcam interface
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
 * Author: Wind Yuan <feng.yuan@intel.com>
 */

/*! \file gstxcaminterface.h
 * \brief Gstreamer XCam 3A interface
 *
 */

#ifndef GST_XCAM_INTERFACE_H
#define GST_XCAM_INTERFACE_H

#include <gst/gst.h>
#include <linux/videodev2.h>
#include <base/xcam_3a_types.h>


G_BEGIN_DECLS

/*! \brief Get GST interface type of XCam 3A interface
 *
 * \return    GType    returned by g_type_register_static()
 */
#define GST_TYPE_XCAM_3A_IF (gst_xcam_3a_interface_get_type ())

/*! \brief Get GST XCam 3A handle.
 * See usage of struct _GstXCam3AInterface.
 *
 * \return    XCam 3A handle of _GstXCam3A * type
 */
#define GST_XCAM_3A(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST ((obj), GST_TYPE_XCAM_3A_IF, GstXCam3A))

/*! \brief Get GST XCam 3A interface
 *
 * See usage of struct _GstXCam3AInterface.
 *
 * \param[in]    Xcam 3A handle
 * \return       GstXCam3AInterface*
 */
#define GST_XCAM_3A_GET_INTERFACE(inst) \
  (G_TYPE_INSTANCE_GET_INTERFACE ((inst), GST_TYPE_XCAM_3A_IF, GstXCam3AInterface))

typedef struct _GstXCam3A GstXCam3A;
typedef struct _GstXCam3AInterface GstXCam3AInterface;

/*! \brief XCam 3A Interface
 *
 * Usage:
 * - GstXCam3A *xcam = GST_XCAM_3A (xcamsrc);
 * - GstXCam3AInterface *xcam_interface = GST_XCAM_3A_GET_INTERFACE (xcam);
 * - ret = xcam_interface->set_exposure_mode(xcam, XCAM_AE_MODE_AUTO);
 */
struct _GstXCam3AInterface {
    GTypeInterface base; /*!< inherent from GTypeInterface */

    /*! \brief Set white balance mode.
     * See xcam_3a_set_whitebalance_mode().
     *
     * \param[in,out]    xcam    XCam handle
     * \param[in]        mode    white balance mode
     * return            0 on success; -1 on error (parameter error)
     */
    gboolean (* set_white_balance_mode)         (GstXCam3A *xcam, XCamAwbMode mode);

    /*! \brief set AWB speed.
     * see xcam_3a_set_awb_speed().
     *
     * \param[in,out]    xcam    XCam handle
     * \param[in,out]    speed   AWB speed; speed meaturement will consider later
     * return            0 on success; -1 on error
     */
    gboolean (* set_awb_speed)                  (GstXCam3A *xcam, double speed);

    /*! \brief Set white balance temperature range.
     * see xcam_3a_set_awb_color_temperature_range().
     *
     * \param[in]    cct_min      0 < cct_min <= cct_max <= 10000; if 0, disable cct range
     * \param[in]    cct_max      0 < cct_min <= cct_max <= 10000; if 0, disable cct range
     * \return       0 on success; -1 on error
     *
     * Usage:
     *
     * - Enable:
     *     1. set_white_balance_mode(%XCAM_AWB_MODE_MANUAL)
     *     2. set_wb_color_temperature_range
     * - Disable:
     *     set_white_balance_mode(%XCAM_AWB_MODE_AUTO)
     *
     */
    gboolean (* set_wb_color_temperature_range) (GstXCam3A *xcam, guint cct_min, guint cct_max);

    /*! \brief Set manual white balance gain.
     * see xcam_3a_set_wb_manual_gain().
     *
     * \param[in,out]    xcam    XCam handle
     * \param[in]        gr      GR channel
     * \param[in]        r       R channel
     * \param[in]        b       B channel
     * \param[in]        gb      GB channel
     *
     * Usage:
     *
     * - Enable:
     *     1. need gr, r, b, gb => gain value [0.1~4.0];
     *     2. set_white_balance_mode(xcam, XCAM_AWB_MODE_NOT_SET)
     * - Disable:
     *     1. need set gr=0, r=0, b=0, gb=0;
     *     2. set_white_balance_mode(xcam, mode);  mode != XCAM_AWB_MODE_NOT_SET
     */
    gboolean (* set_manual_wb_gain)             (GstXCam3A *xcam, double gr, double r, double b, double gb);


    /*! \brief set exposure mode.
     * see xcam_3a_set_exposure_mode().
     *
     * \param[in,out]    xcam    XCam handle
     * \param[in]        mode    choose from XCAM_AE_MODE_AUTO and XCAM_AE_MODE_MANUAL; others not supported
     */
    gboolean (* set_exposure_mode)              (GstXCam3A *xcam, XCamAeMode mode);

    /*! \brief set AE metering mode.
     * see xcam_3a_set_ae_metering_mode().
     *
     * \param[in,out]    xcam    XCam handle
     * \param[in]        mode    XCAM_AE_METERING_MODE_AUTO, default
     *                           XCAM_AE_METERING_MODE_SPOT, need set spot window by set_exposure_window
     *                           XCAM_AE_METERING_MODE_CENTER,  more weight in center
     *                           XCAM_AE_METERING_MODE_WEIGHTED_WINDOW,  weighted multi metering window
     */
    gboolean (* set_ae_metering_mode)           (GstXCam3A *xcam, XCamAeMeteringMode mode);

    /* \brief set exposure window.
     * see xcam_3a_set_ae_window().
     *
     * \param[in,out]    xcam      XCam handle
     * \param[in]        window    the area to set exposure with. x_end > x_start AND y_end > y_start; only ONE window can be set
     * \param[in]        count     the number of metering window
     *
     * Usage
     * - Enable:
     *     set_ae_metering_mode(@xcam, %XCAM_AE_METERING_MODE_SPOT)
     * - Disable:
     *     set_ae_metering_mode(@xcam, @mode); #mode != %XCAM_AE_METERING_MODE_SPOT
     */
    gboolean (* set_exposure_window)            (GstXCam3A *xcam, XCam3AWindow *window, guint8 count);

    /*! \brief set exposure value offset.
     * see xcam_3a_set_ae_value_shift().
     *
     * \param[in,out]    xcam        XCam handle
     * \param[in]        ev_offset   -4.0 <= ev_offset <= 4.0; default 0.0
     */
    gboolean (* set_exposure_value_offset)      (GstXCam3A *xcam, double ev_offset);

    /*! \brief set  AE speed.
     * see xcam_3a_set_ae_speed().
     *
     * \param[in,out]    xcam        XCam handle
     * \param[in]        speed       AE speed
     */
    gboolean (* set_ae_speed)                   (GstXCam3A *xcam, double speed);

    /*! \brief set exposure flicker mode.
     * see xcam_3a_set_ae_flicker_mode().
     *
     * \param[in,out]    xcam        XCam handle
     * \param[in]        flicker     XCAM_AE_FLICKER_MODE_AUTO, default
     *                               XCAM_AE_FLICKER_MODE_50HZ
     *                               XCAM_AE_FLICKER_MODE_60HZ
     *                               XCAM_AE_FLICKER_MODE_OFF, outside
     */
    gboolean (*set_exposure_flicker_mode)       (GstXCam3A *xcam, XCamFlickerMode flicker);

    /*! \brief get exposure flicker mode.
     * see xcam_3a_get_ae_flicker_mode().
     *
     * \param[in,out]    xcam                XCam handle
     * \return           XCamFlickerMode     XCAM_AE_FLICKER_MODE_AUTO, default
     *                                       XCAM_AE_FLICKER_MODE_50HZ
     *                                       XCAM_AE_FLICKER_MODE_60HZ
     *                                       XCAM_AE_FLICKER_MODE_OFF, outside
     */
    XCamFlickerMode (*get_exposure_flicker_mode)      (GstXCam3A *xcam);

    /*! \brief get current exposure time.
     * see xcam_3a_get_current_exposure_time().
     *
     * \param[in,out]    xcam        XCam handle
     * \return           current exposure time in microsecond, if return -1, means xcam is not started
     */
    gint64   (* get_current_exposure_time)      (GstXCam3A *xcam);

    /*! \brief get current analog gain.
     * see xcam_3a_get_current_analog_gain().
     *
     * \param[in,out]    xcam        XCam handle
     * \return            current analog gain as multiplier. If return < 0.0 OR return < 1.0,  xcam is not started.
     */
    double   (* get_current_analog_gain)        (GstXCam3A *xcam);

    /*! \brief set manual exposure time
     *
     * \param[in,out]    xcam          XCam handle
     * \param[in]        time_in_us    exposure time
     *
     * Usage:
     * - Enable:
     *      set time_in_us, 0 < time_in_us < 1/fps
     * - Disable:
     *     time_in_us = 0
     */
    gboolean (* set_manual_exposure_time)       (GstXCam3A *xcam, gint64 time_in_us);

    /*! \brief set manual analog gain.
     * see  xcam_3a_set_ae_manual_analog_gain().
     *
     * \param[in,out]    xcam          XCam handle
     * \param[in]        gain          analog gain
     *
     * Usage:
     * - Enable:
     *     set @gain value, 1.0 < @gain
     * - Disable:
     *     set @gain = 0.0
     */
    gboolean (* set_manual_analog_gain)         (GstXCam3A *xcam, double gain);

    /*! \brief set aperture.
     * see xcam_3a_set_ae_set_aperture().
     *
     * \param[in,out]    xcam            XCam3A handle
     * \param[in]        fn              AE aperture fn
     * \return           bool            0 on success
     */
    gboolean (* set_aperture)                   (GstXCam3A *xcam, double fn);

    /*! \brief set max analog gain.
     * see xcam_3a_set_ae_max_analog_gain().
     *
     * \param[in,out]    xcam            XCam3A handle
     * \param[in]        max_gain        max analog gain
     * \return           gboolen         0 on success
     */
    gboolean (* set_max_analog_gain)            (GstXCam3A *xcam, double max_gain);

    /*! \brief get max analog gain.
     * see xcam_3a_get_ae_max_analog_gain().
     *
     * \param[in,out]    xcam            XCam3A handle
     * \return           max_gain        max analog gain
     */
    double   (* get_max_analog_gain)            (GstXCam3A *xcam);

    /*!
     * \brief set AE time range
     *
     * \param[in,out]    xcam            XCam3A handle
     * \param[in]        min_time_in_us  min time
     * \param[in]        max_time_in_us  max time
     * \return           XCam3AStatus    0 on success
     */
    gboolean (* set_exposure_time_range)        (GstXCam3A *xcam, gint64 min_time_in_us, gint64 max_time_in_us);

    /*!
     * \brief XCam3A get AE time range.
     * Range in [0 ~ 1000000/fps] micro-seconds. see xcam_3a_set_ae_time_range().
     *
     * \param[in,out]    xcam            XCam3A handle
     * \param[out]       min_time_in_us  min time
     * \param[out]       max_time_in_us  max time
     * \return           bool            0 on success
     */
    gboolean (* get_exposure_time_range)        (GstXCam3A *xcam, gint64 *min_time_in_us, gint64 *max_time_in_us);

    /*! \brief set DVS.
     *  digital video stabilization. see xcam_3a_enable_dvs().
     *
     * \param[in,out]    xcam            XCam3A handle
     * \param[in]        enable          enable/disable
     * \return           bool            0 on success
     */
    gboolean (* set_dvs)                        (GstXCam3A *xcam, gboolean enable);

    /*! \brief set noice reduction level to BNR and YNR.
     * see xcam_3a_set_noise_reduction_level().
     *
     * \param[in,out]    xcam            XCam3A handle
     * \param[in]        level           control BNR/YNR gain. 0 <= level <= 255; default level: 128
     * \return           bool            0 on success
     */
    gboolean (*set_noise_reduction_level)       (GstXCam3A *xcam, guint8 level);

    /*! \brief set temporal noice reduction level.
     * see xcam_3a_set_temporal_noise_reduction_level().
     *
     * \param[in,out]    xcam            XCam3A handle
     * \param[in]        level           control TNR gain. 0 <= level <= 255; default level: 128
     * \param[in]        mode            TNR filter mode  0: disable, 1: YUV mode, 2: RGB mode
     * \return           bool            0 on success
     */
    gboolean (*set_temporal_noise_reduction_level) (GstXCam3A *xcam, guint8 level, gint8 mode);

    /*!
     * \brief set gamma table.
     * see xcam_3a_set_set_gamma_table().
     *
     * \param[in,out]    xcam          XCam3A handle
     * \param[in]        r_table         red color gamma table
     * \param[in]        g_table         green color gamma table
     * \param[in]        b_table         blue color gamma table
     * \return           bool            0 on success
     *
     * Restriction:
     *     1. can't co-work with manual brightness and contrast,
     *     2. table size = 256, and values in [0.0~1.0], e.g 0.0, 1.0/256,  2.0/256 ... 255.0/256
     *
     * Usage:
     * - to Disable:
     *     r_table = NULL && g_table = NULL && b_table=NULL
     */
    gboolean (* set_gamma_table)                (GstXCam3A *xcam, double *r_table, double *g_table, double *b_table);

    /*!
     * \brief enable/disable gbce.
     * see xcam_3a_enable_gbce().
     *
     * \param[in,out]    xcam          XCam3A handle
     * \param[in]        enable        enable/disable, i.e. TRUE to enable GBCE and otherwise disable GBCE.
     * \return           bool          0 on success
     */
    gboolean (* set_gbce)                       (GstXCam3A *xcam, gboolean enable);

    /*!
     * \brief set manual brightness.
     * see xcam_3a_set_manual_brightness().
     *
     * \param[in,out]    xcam            XCam3A handle
     * \param[in]        value           manual brightness, 0 <= value <= 255; default:128
     * \return           bool            0 on success    */
    gboolean (* set_manual_brightness)          (GstXCam3A *xcam, guint8 value);

    /*!
     * \brief set manual contrast.
     * see xcam_3a_set_manual_contrast().
     *
     * \param[in,out]    xcam            XCam3A handle
     * \param[in]        value           manual contrast, 0 <= value <= 255; default:128
     * \return           bool            0 on success    */
    gboolean (* set_manual_contrast)            (GstXCam3A *xcam, guint8 value);

    /*!
     * \brief set manual hue.
     * see xcam_3a_set_manual_hue().
     *
     * \param[in,out]    xcam            XCam3A handle
     * \param[in]        value           manual hue, 0 <= value <= 255; default:128
     * \return           bool            0 on success    */
    gboolean (* set_manual_hue)                 (GstXCam3A *xcam, guint8 value);

    /*!
     * \brief set manual saturation.
     * see xcam_3a_set_manual_saturation().
     *
     * \param[in,out]    xcam            XCam3A handle
     * \param[in]        value           manual saturation, 0 <= value <= 255; default:128
     * \return           bool            0 on success    */
    gboolean (* set_manual_saturation)          (GstXCam3A *xcam, guint8 value);

    /*!
     * \brief set manual sharpness.
     * see xcam_3a_set_manual_sharpness().
     *
     * \param[in,out]    xcam            XCam3A handle
     * \param[in]        value           manual sharpness, 0 <= value <= 255; default:128
     * \return           bool            0 on success    */
    gboolean (* set_manual_sharpness)           (GstXCam3A *xcam, guint8 value);

    /* IR-cut */
    /*!
     * \brief enable/disable night mode.
     * see xcam_3a_enable_night_mode().
     *
     * \param[in,out]    xcam          XCam3A handle
     * \param[in]        enable        enable/disable, i.e. TRUE to enable night mode and otherwise disable night mode.
     * \return           bool          0 on success
     */
    gboolean (* set_night_mode)                 (GstXCam3A *xcam, gboolean enable);

    /*!
     * \brief set HDR mode.
     *
     * \param[in,out]    xcam          XCam3A handle
     * \param[in]        mode          0: disable, 1: HDR in RGB color space, 2: HDR in LAB color space
     * \return           bool          0 on success
     */
    gboolean (* set_hdr_mode)                   (GstXCam3A *xcam, guint8 mode);

    /*!
     * \brief set denoise mode.
     *
     * \param[in,out]    xcam          XCam3A handle
     * \param[in]        mode          bit mask to enable/disable denoise functions
     *                                 each bit controls a specific denoise function, 0: disable, 1: enable
     *                                   bit 0: simple noise reduction
     *                                   bit 1: bilateral noise reduction
     *                                   bit 2: luminance noise reduction and edge enhancement
     *                                   bit 3: bayer noise reduction
     *                                   bit 4: advanced bayer noise reduction
     * \return           bool          0 on success
     */
    gboolean (* set_denoise_mode)               (GstXCam3A *xcam, guint32 mode);

    /*!
     * \brief set gamma mode.
     *
     * \param[in,out]    xcam          XCam3A handle
     * \param[in]        enable        true: enable, false: disable
     * \return           bool          0 on success
     */
    gboolean (* set_gamma_mode)                 (GstXCam3A *xcam, gboolean enable);

    /*!
     * \brief set dpc mode.
     *
     * \param[in,out]    xcam          XCam3A handle
     * \param[in]        enable        true: enable, false: disable
     * \return           bool          0 on success
     */
    gboolean (* set_dpc_mode)                   (GstXCam3A *xcam, gboolean enable);

    /*!
     * \brief set tone mapping mode.
     *
     * \param[in,out]    xcam          XCam3A handle
     * \param[in]        enable        true: enable, false: disable
     * \return           bool          0 on success
     */
    gboolean (* set_tonemapping_mode)           (GstXCam3A *xcam, gboolean enable);

    /*!
     * \brief set retinex mode.
     *
     * \param[in,out]    xcam          XCam3A handle
     * \param[in]        enable        true: enable, false: disable
     * \return           bool          0 on success
     */
    gboolean (* set_retinex_mode)           (GstXCam3A *xcam, gboolean enable);


};

/*! \brief Get GST interface type of XCam 3A interface.
 * will try to register GsXcam3AInterface with
 * g_type_register_static() if not done so yet, and in turn return the
 * interface type it returns.
 *
 * \return    GType    XCam 3A interface type returned by g_type_register_static()
 */
GType
gst_xcam_3a_interface_get_type (void);

G_END_DECLS

#endif /* GST_XCAM_INTERFACE_H */
