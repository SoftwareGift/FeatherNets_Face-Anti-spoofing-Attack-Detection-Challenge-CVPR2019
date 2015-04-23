/*
 * gstxcamsrc.h - gst xcamsrc plugin
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

#ifndef __GST_XCAMSRC_H__
#define __GST_XCAMSRC_H__

#include <gst/gst.h>
#include <gst/base/gstpushsrc.h>
#include <linux/videodev2.h>
#include <xcam_defs.h>

#define DEFAULT_BLOCKSIZE   1843200
#define CAPTURE_DEVICE_STILL  "/dev/video0"
#define CAPTURE_DEVICE_VIDEO  "/dev/video3"
#define DEFAULT_EVENT_DEVICE    "/dev/v4l-subdev6"
#define DEFAULT_CPF_FILE_NAME   "/etc/atomisp/imx185.cpf"

#define DEFAULT_PROP_DEVICE_NAME    "/dev/video3"
#define DEFAULT_PROP_SENSOR     0
#define DEFAULT_PROP_CAPTURE_MODE    0
#define DEFAULT_PROP_IO_MODE        4
#define DEFAULT_PROP_BUFFERCOUNT    8
#define DEFAULT_PROP_FPSN       25
#define DEFAULT_PROP_FPSD       1
#define DEFAULT_PROP_WIDTH      1920
#define DEFAULT_PROP_HEIGHT     1080
#define DEFAULT_PROP_PIXELFORMAT    V4L2_PIX_FMT_NV12 //420 instead of 0
#define DEFAULT_PROP_FIELD      V4L2_FIELD_NONE // 0
#define DEFAULT_PROP_BYTESPERLINE   3840

#define V4L2_CAPTURE_MODE_STILL 0x2000
#define V4L2_CAPTURE_MODE_VIDEO 0x4000
#define V4L2_CAPTURE_MODE_PREVIEW 0x8000

G_BEGIN_DECLS

/* #defines don't like whitespacey bits */
#define GST_TYPE_XCAMSRC \
  (gst_xcamsrc_get_type())
#define GST_XCAMSRC(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_XCAMSRC,Gstxcamsrc))
#define GST_XCAMSRC_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_XCAMSRC,GstxcamsrcClass))
#define GST_IS_XCAMSRC(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_XCAMSRC))
#define GST_IS_XCAMSRC_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_XCAMSRC))
#define GST_XCAMSRC_CAST(obj)   ((Gstxcamsrc *) obj)

typedef enum {
    ISP_IMAGE_PROCESSOR = 0,
    CL_IMAGE_PROCESSOR,
} ImageProcessorType;

typedef enum {
    SIMPLE_ANALYZER = 0,
    AIQ_ANALYZER,
} AnalyzerType;

typedef struct _Gstxcamsrc      Gstxcamsrc;
typedef struct _GstxcamsrcClass GstxcamsrcClass;

struct _Gstxcamsrc
{
    GstPushSrc pushsrc;

    GstBufferPool *pool;

    guint buf_count;

    guint _fps_n;
    guint _fps_d;

    guint width;
    guint height;
    guint pixelformat;
    enum v4l2_field field;
    guint bytes_perline;
    gint sensor_id;
    guint capture_mode;
    enum v4l2_memory mem_type;
    ImageProcessorType image_processor_type;
    AnalyzerType analyzer_type;

    GstClockTime duration;
    GstClockTime ctrl_time;

    guint64 offset;

};

struct _GstxcamsrcClass
{
    GstPushSrcClass parent_class;
};

GType gst_xcamsrc_get_type (void);

G_END_DECLS

#endif /* __GST_XCAMSRC_H__ */
