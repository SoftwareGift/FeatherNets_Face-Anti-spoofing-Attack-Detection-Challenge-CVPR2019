/*
 * test-image-stabilization.cpp - test image stabilization
 *
 *  Copyright (c) 2016 Intel Corporation
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
 * Author: Zong Wei <wei.zong@intel.com>
 */

#include "xcam_utils.h"

#include <unistd.h>
#include <getopt.h>
#include <string>
#include "plugins/smart/dvs/libdvs/stabilizer.h"

using namespace std;
using namespace cv;

void usage(const char* arg0)
{
    printf ("Usage:\n"
            "%s --input file --output file\n"
            "\t--input            input video\n"
            "\t--output           output video\n"
            "\t--enable-twopass   two pass stabilization\n"
            "\t--enable-deblur    do deblur on output video\n"
            "\t--wobble-suppress  do wobble suppress\n"
            "\t--save             save file or not, default: true\n"
            "\t--help             usage\n",
            arg0);
}

int main(int argc, char *argv[])
{
    char inputPath[XCAM_MAX_STR_SIZE] = {0};
    char outputPath[XCAM_MAX_STR_SIZE] = {0};
    bool enableTwoPass = false;
    bool enableDeblur = false;
    bool wobbleSuppress = false;
    bool saveOutput = true;

    const struct option long_opts[] = {
        {"input", required_argument, NULL, 'i'},
        {"output", required_argument, NULL, 'o'},
        {"enable-twopass", required_argument, NULL, 'p'},
        {"enable-deblur", required_argument, NULL, 'd'},
        {"wobble-suppress", required_argument, NULL, 'w'},
        {"save", required_argument, NULL, 's'},
        {"help", no_argument, NULL, 'e'},
        {NULL, 0, NULL, 0},
    };

    int opt = -1;
    while ((opt = getopt_long(argc, argv, "", long_opts, NULL)) != -1) {
        switch (opt) {
        case 'i':
            strncpy (inputPath, optarg, XCAM_MAX_STR_SIZE);
            break;
        case 'o':
            strncpy (outputPath, optarg, XCAM_MAX_STR_SIZE);
            break;
        case 'p':
            enableTwoPass = (strcasecmp (optarg, "false") == 0 ? false : true);;
            break;
        case 'd':
            enableDeblur = (strcasecmp (optarg, "false") == 0 ? false : true);
            break;
        case 'w':
            wobbleSuppress = (strcasecmp (optarg, "false") == 0 ? false : true);
            break;
        case 's':
            saveOutput = (strcasecmp (optarg, "false") == 0 ? false : true);
            break;
        case 'e':
            usage (argv[0]);
            return -1;
        default:
            XCAM_LOG_ERROR ("getopt_long return unknown value:%c", opt);
            usage (argv[0]);
            return -1;
        }
    }

    if (optind < argc || argc < 2) {
        XCAM_LOG_ERROR ("unknown option %s", argv[optind]);
        usage (argv[0]);
        return -1;
    }

    printf ("Description----------------\n");
    printf ("input file:\t%s\n", inputPath);
    printf ("output file:\t%s\n", outputPath);
    printf ("enable two pass stabilizer:\t%s\n", enableTwoPass ? "true" : "false");
    printf ("enable deblur:\t%s\n", enableDeblur ? "true" : "false");
    printf ("enable wobble suppress:\t%s\n", wobbleSuppress ? "true" : "false");
    printf ("save file:\t%s\n", saveOutput ? "true" : "false");
    printf ("---------------------------\n");

    Ptr<VideoStabilizer> dvs = makePtr<VideoStabilizer>(enableTwoPass, wobbleSuppress, enableDeblur);
    Ptr<StabilizerBase> stabilizer = dvs->stabilizer();

    Ptr<VideoFileSource> source = makePtr<VideoFileSource>(inputPath);
    stabilizer->setFrameSource(source);

    int outputFps = source->fps();
    Size frameSize = Size(source->width(), source->height());
    cout << "frame count (rough): " << source->count() << endl;
    cout << "output FPS: " << outputFps << endl;
    cout << "frame size: " << source->width() << "x" << source->height() << endl;

    // stabilizer configuration
    dvs->configFeatureDetector(1000, 15.0f);
    dvs->configMotionFilter(15, 10.0f);

    // start to run
    Mat stabilizedFrame, croppedStabilizedFrame;
    int nframes = 0;

    while (!(stabilizedFrame = dvs->nextFrame()).empty())
    {
        nframes++;
        cout << nframes << endl;

        // doing cropping here
        croppedStabilizedFrame = dvs->cropVideoFrame(stabilizedFrame);

        if (saveOutput) {
            if (!dvs->writer_.isOpened()) {
                dvs->writer_.open(outputPath, VideoWriter::fourcc('X', '2', '6', '4'),
                                  outputFps, dvs->trimedVideoSize(frameSize));
            }
            dvs->writer_.write(croppedStabilizedFrame);
        }

        imshow("stabilizedFrame", croppedStabilizedFrame);
        char key = static_cast<char>(waitKey(3));
        if (key == 27) {
            cout << endl;
            break;
        }
    }

    return 0;
}
