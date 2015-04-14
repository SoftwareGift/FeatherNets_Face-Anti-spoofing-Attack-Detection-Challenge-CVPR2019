#!/bin/bash

gst-launch-1.0 xcamsrc sensor-id=3 capture-mode=1 io-mode=4 ! video/x-raw, format=NV12, width=1920, height=1080, framerate=30/1 ! queue ! vaapiencode_h264 ! fakesink
