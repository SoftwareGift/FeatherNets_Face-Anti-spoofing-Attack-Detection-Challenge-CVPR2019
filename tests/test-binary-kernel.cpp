/*
 * test-binary-kernel.cpp - Compile the source kernel into binary kernel
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
 * Author: Yinhang Liu <yinhangx.liu@intel.com>
 */

#include "test_common.h"
#include "test_inline.h"
#include "file_handle.h"
#include "ocl/cl_device.h"
#include "ocl/cl_context.h"
#include "ocl/cl_kernel.h"
#include <getopt.h>

using namespace XCam;

static void
print_help (const char *arg0)
{
    printf ("Usage: %s --src-kernel <source-kernel> --bin-kernel <binary-kernel> --kernel-name <kernel-name>\n"
            "\t --src-kernel   specify source kernel path\n"
            "\t --bin-kernel   specify binary kernel path\n"
            "\t --kernel-name  specify kernel name\n"
            "\t --help         help\n"
            , arg0);
}

#define FAILED_STATEMENT {                         \
        if (kernel_body) xcam_free (kernel_body);  \
        if (kernel_name) xcam_free (kernel_name);  \
        if (program_binaries) xcam_free (program_binaries); \
        return -1; }

int main (int argc, char *argv[])
{
    char *src_path = NULL, *bin_path = NULL;
    size_t src_size = 0;
    size_t bin_size = 0;
    char *kernel_name = NULL;
    char *kernel_body = NULL;
    uint8_t *program_binaries = NULL;
    FileHandle src_file, bin_file;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    const struct option long_opts [] = {
        {"src-kernel", required_argument, NULL, 's'},
        {"bin-kernel", required_argument, NULL, 'b'},
        {"kernel-name", required_argument, NULL, 'n'},
        {"help", no_argument, NULL, 'h'},
        {NULL, 0, NULL, 0}
    };

    int opt = 0;
    while ((opt = getopt_long (argc, argv, "", long_opts, NULL)) != -1) {
        switch (opt) {
        case 's':
            src_path = optarg;
            break;
        case 'b':
            bin_path = optarg;
            break;
        case 'n':
            kernel_name = strndup (optarg, 1024);
            break;
        case 'h':
            print_help (argv[0]);
            return 0;

        default:
            print_help (argv[0]);
            return -1;
        }
    }

    if (!src_path || !bin_path) {
        XCAM_LOG_ERROR ("path of source/binary kernel is null");
        return -1;
    }
    if (!kernel_name) {
        XCAM_LOG_ERROR ("kernel name is null");
        return -1;
    }

    if (src_file.open (src_path, "r") != XCAM_RETURN_NO_ERROR ||
            bin_file.open (bin_path, "wb")  != XCAM_RETURN_NO_ERROR) {
        XCAM_LOG_ERROR ("open source/binary kernel failed");
        return -1;
    }

    ret = src_file.get_file_size (src_size);
    CHECK_STATEMENT (ret, FAILED_STATEMENT, "get source sizes from %s failed", src_path);

    kernel_body = (char *) xcam_malloc0 (sizeof (char) * (src_size + 1));
    XCAM_ASSERT(kernel_body);

    src_file.read_file (kernel_body, src_size);
    CHECK_STATEMENT (ret, FAILED_STATEMENT, "read source from %s failed", src_path);
    kernel_body[src_size] = '\0';

    SmartPtr<CLContext> context;
    context = CLDevice::instance ()->get_context ();
    SmartPtr<CLKernel> kernel = new CLKernel (context, kernel_name);
    kernel->load_from_source (kernel_body, strlen (kernel_body), &program_binaries, &bin_size);

    ret = bin_file.write_file (program_binaries, bin_size);
    CHECK_STATEMENT (ret, FAILED_STATEMENT, "write binary to %s failed", bin_path);

    xcam_free (kernel_name);
    xcam_free (kernel_body);
    xcam_free (program_binaries);
    return 0;
}
