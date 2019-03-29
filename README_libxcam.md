## libXCam

Copyright (C) 2014-2018 Intel Corporation

libxcam core source code under the terms of Apache License, Version 2.0

#### Description:
libXCam is a project for extended camera features and focus on image
quality improvement and video analysis. There are lots features supported
in image pre-processing, image post-processing and smart analysis. This
library makes GPU/CPU/ISP working together to improve image quality.
OpenCL is used to improve performance in different platforms.

#### Features:
  * Image processing features.
    - Advanced features.
      - Automotive surround view(360) stitching (OpenCL/CPU/GLES).
         - Support bowl view 3D model stitching by 4 video input.
         - Enable geometry remap for WFoV camera calibration(intrinsic and extrinsic data).
         - Quality and performance improved (OpenCL/CPU/GLES).
         - CPU version upstreamed into AOSP for automotive surround view.
         - Enable Vulkan to improve performance.
      - 360 video stitching (Equirectangular mode via OpenCL).
        - Support 2-fisheye (>180 degree) video stream stitching.
        - Performance and quality improved.
      - Digital Video Stabilization:
        - OpenCV feature-matched based video stabilization.
        - gyroscope 3-DoF (orientation) based video stabilization.
      - Blender: multi-band blender (OpenCL/CPU/GLES).
      - Noise reduction (OpenCL).
        - adaptive NR based on wavelet-haar and Bayersian shrinkage.
        - 3D-NR with inter-block and intra-block reference.
        - wavelet-hat NR (obsolete).
      - Wide dynamic range (WDR) (OpenCL).
        - histogram adjustment tone-mapping.
        - gaussian-based tone-mapping (obsolete).
      - Fog removal: retinex and dark channel prior algorithm (OpenCL).
        - dark channel prior algorithm based defog.
        - multi-scale retinex based defog (obsolete).
    - Basic pipeline from bayer to YUV/RGB format (OpenCL / AtomISP).
      - Gamma correction, MACC, color space, demosaicing, simple bilateral
        noise reduction, edge enhancement and temporal noise reduction.
    - 3A features.
      - Auto whitebalance, auto exposure, auto focus, black level correction,
        color correction, 3a-statistics calculation.
  * Support 3rd party 3A lib which can be loaded dynamically.
       - hybrid 3a plugin.
  * Support 3a analysis tuning framework for different features.
  * Support smart analysis framework.
       - Face detection interface/plugin.
  * Enable gstreamer plugin.
       - xcamsrc, capture from usb/isp camera, process 3a/basic/advanced features.
       - xcamfilter, improve image quality by advanced features and smart analysis.

#### Prerequisite:
  * install gcc/g++, automake, autoconf, libtool, gawk, pkg-config
  * Linux kernel > 3.10
  * install ocl-icd-dev, ocl-icd-opencl-dev
  * If --enable-libcl, need compile ocl driver <https://www.freedesktop.org/wiki/Software/Beignet/>
  * If --enable-opencv, suggest opencv versions [3.0.0 - 3.4.3]<http://opencv.org> (or: <https://github.com/opencv/opencv/wiki>)
  * If --enable-gst, need install libgstreamer1.0-dev, libgstreamer-plugins-base1.0-dev
  * If --enable-aiq, need get ia_imaging lib which we don't support.
  * If --enable-render, need compile OpenSceneGraph library with configure option "-DOSG_WINDOWING_SYSTEM=X11" <https://github.com/openscenegraph/OpenSceneGraph.git>
  * If --enable-gles, need to install mesa-3d library
  * If --enable-vulkan, need to install mesa-3d library

#### Building and installing:
  * Environment variable settings<BR>
    For different --prefix options, the environment variables may be different. Please set the environment variable according to the actual situation.<BR>
    --prefix=/usr/local:

        export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH
        export GST_PLUGIN_PATH=/usr/local/lib/gstreamer-1.0:$GST_PLUGIN_PATH

    --prefix=/usr:

        export LD_LIBRARY_PATH=/usr/lib/:$LD_LIBRARY_PATH
        export GST_PLUGIN_PATH=/usr/lib/gstreamer-1.0:$GST_PLUGIN_PATH

  * $ ./autogen.sh [options]

        --prefix=PREFIX         install architecture-independent files in PREFIX [default=/usr/local]
        --enable-debug          enable debug, [default=no]
        --enable-profiling      enable profiling, [default=no]
        --enable-drm            enable drm buffer, [default=no]
        --enable-aiq            enable Aiq 3A algorithm build, [default=no]
        --enable-gst            enable gstreamer plugin build, [default=no]
        --enable-libcl          enable libcl image processor, [default=yes]
        --enable-opencv         enable opencv library, [default=no]
        --enable-capi           enable libxcam-capi library, [default=no]
        --enable-docs           build Doxygen documentation [default=no]
        --enable-3alib          enable 3A lib build, [default=no]
        --enable-smartlib       enable smart analysis lib build, [default=no]
        --enable-gles           enable gles, [default=no]
        --enable-vulkan         enable vulkan, [default=no]
        --enable-render         enable 3D texture render, [default=no]

    For example:

        $ ./autogen.sh --prefix=/usr --enable-gst --enable-libcl --enable-opencv \
          --enable-smartlib --enable-profiling --enable-gles --enable-render

  * $ make
  * $ sudo make install

#### Testing:
  * For detailed test cases, please refer to:<BR>
    <https://github.com/intel/libxcam/wiki/Tests>

#### Reporting Bugs:
  * Bugs or suggestions can be reported on the github issues page:<BR>
    <https://github.com/intel/libxcam/issues>
  * Security issues, please send email to feng.yuan@intel.com directly

#### Mailing list
  * To post a message to all the list members, please send email to libxcam@lists.01.org.
  * To register libxcam public maillist, please go to:<BR>
    <https://lists.01.org/mailman/listinfo/libxcam>

#### Maintainer:
  * Wind Yuan <feng.yuan@intel.com>

#### Contributors: (orders by first name)
  * Andrey Parfenov <a1994ndrey@gmail.com>
  * Fei Wang <feix.w.wang@intel.com>
  * Jia Meng <jia.meng@intel.com>
  * John Ye <john.ye@intel.com>
  * Juan Zhao <juan.j.zhao@intel.com>
  * Junkai Wu <junkai.wu@intel.com>
  * Sameer Kibey <sameer.kibey@intel.com>
  * Shincy Tu <shincy.tu@intel.com>
  * Wei Zong <wei.zong@intel.com>
  * Yan Zhang <yan.y.zhang@intel.com>
  * Yao Wang <yao.y.wang@intel.com>
  * Yinhang Liu <yinhangx.liu@intel.com>
