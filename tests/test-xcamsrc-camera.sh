#!/bin/bash

gst-launch-1.0 xcamsrc sensor=0 capturemode=0x4000 memtype=4 buffercount=8 fpsn=25 fpsd=1 width=1920 height=1080 pixelformat=0 field=0 bytesperline=3840 ! video/x-raw, format=NV12, width=1920, height=1080, framerate=30/1 ! queue ! vaapiencode_h264 ! fakesink
