## libXCam

Copyright (C) 2014-2017 Intel Corporation

libxcam core source code under the terms of Apache License, Version 2.0

#### Description:
libXCam is a project for extended camera features and focus on image
quality improvement and video analysis. There are lots features supported
in image pre-processing, image post-processing and smart analysis. This
library makes GPU/CPU/ISP working together to improve image quality.
OpenCL is used to improve performance in different platforms.

#### Features:
  * Image processing features.
   - Basic features
       Gamma correction, MACC, color space, demosaicing, noise reduction,
       edge enhancement.
    - Advanced features
       - Noise reduction: simple bilateral NR, temporal NR, wavelet NR,
         adaptive NR with Bayesian shrinkage, 3D-NR.
       - WDR: gaussian-based tone-mapping, histogram adjustment tone-mapping
       - Fog removal: retinex and dark channel prior algorithm.
       - Blender: multi-band blender and fisheye geometry map correction.
       - Image stiching: feature detect and match to generate 360 degree
         panorama photography by stitching multiple neighbor images
       - DVS: digital video stabilization based on feature detect, optical flow
         and motion model estimation.
     - 3A features
       Auto whitebalance, auto exposure, auto focus, black level correction,
       color correction, 3a-statistics calculation.
  * Capture features support memory type of mmap and dma buffer.
  * Support 3rd party 3A lib which can be loaded dynamically.
       - hybrid 3a plugin
  * Image processing based on both ISP and Open CL.
  * Support 3a analysis tuning framework for different features
  * Support smart analysis framework
       - Face detection interface/plugin
  * Support as gstreamer plugin
       - xcamsrc, capture from usb/isp camera, process 3a/basic/advanced features.
       - xcamfilter, improve image quality by advanced features and smart analysis.

#### Prerequisite:
  * install gcc/g++, automake, autoconf, libtool, gawk, pkg-config
  * Linux kernel > 3.10
  * install libdrm-dev
  * install ocl-icd-dev, ocl-icd-opencl-dev
  * If --enable-libcl, need compile ocl driver <https://www.freedesktop.org/wiki/Software/Beignet/>
  * If --enable-opencv, need compile opencv <http://opencv.org> (or: <https://github.com/opencv/opencv/wiki>)
  * If --enable-gst, need install libgstreamer1.0-dev, libgstreamer-plugins-base1.0-dev
  * If --enable-aiq, need get ia_imaging lib which we don't support.

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
        --enable-drm            enable drm buffer, [default=yes]
        --enable-aiq            enable Aiq 3A algorithm build, [default=no]
        --enable-gst            enable gstreamer plugin build, [default=no]
        --enable-libcl          enable libcl image processor, [default=yes]
        --enable-opencv         enable opencv library, [default=no]
        --enable-docs           build Doxygen documentation [default=no]
        --enable-3alib          enable 3A lib build, [default=no]
        --enable-smartlib       enable smart analysis lib build, [default=no]

    For example:

        $ ./autogen.sh --prefix=/usr --enable-3alib --enable-aiq --enable-gst --enable-drm \
          --enable-libcl --enable-opencv --enable-profiling --enable-smartlib

  * $ make
  * $ sudo make install

#### Testing:
  * For detailed test cases, please refer to:<BR>
    <https://github.com/01org/libxcam/wiki/Tests>

#### Reporting Bugs:
  * Bugs or suggestions can be reported on the github issues page:<BR>
    <https://github.com/01org/libxcam/issues>
