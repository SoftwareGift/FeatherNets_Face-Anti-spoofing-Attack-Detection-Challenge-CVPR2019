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
#include "cl_device.h"
#include "cl_context.h"
#include "cl_kernel.h"
#include <unistd.h>
#include <libgen.h>

using namespace XCam;

struct TestFileHandle {
    FILE *fp;
    TestFileHandle ()
        : fp (NULL)
    {}
    ~TestFileHandle ()
    {
        if (fp)
            fclose (fp);
    }
};

static XCamReturn
get_kernel_name(char *input_file, char **kernel_name)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    char *base_name = basename (input_file);
    size_t kernel_name_length = strlen (base_name) - 3;

    *kernel_name = (char *) xcam_malloc0 (sizeof (char) * (kernel_name_length + 1));
    XCAM_ASSERT(*kernel_name);

    strncpy (*kernel_name, base_name, kernel_name_length);

    return ret;
}

static XCamReturn
get_source_sizes (TestFileHandle &file, size_t *source_sizes)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    if (fseek (file.fp, 0 , SEEK_END) != 0)
        goto read_error;

    if ((*source_sizes = ftell (file.fp)) <= 0)
        goto read_error;

    rewind(file.fp);

    return ret;

read_error:
    XCAM_LOG_ERROR ("get source sizes failed");
    return XCAM_RETURN_ERROR_FILE;
}

static XCamReturn
read_source (TestFileHandle &file, char *source, size_t source_sizes)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    if (fread (source, sizeof (char), source_sizes, file.fp) != source_sizes) {
        XCAM_LOG_ERROR ("read source failed, size doesn't match");
        ret = XCAM_RETURN_ERROR_FILE;
    }

    return ret;
}

static XCamReturn
write_binary (TestFileHandle &file, uint8_t *binaries, const size_t binary_sizes)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    if (fwrite (binaries, sizeof (uint8_t), binary_sizes, file.fp) != binary_sizes) {
        XCAM_LOG_ERROR ("write binary failed, size doesn't match");
        ret = XCAM_RETURN_ERROR_FILE;
    }

    return ret;
}

static void
print_help (const char *bin_name)
{
    printf ("Usage: %s -i <source-file> -o <binary-file>\n"
            "\t -i input-file     specify source file path\n"
            "\t -o output-file    specify binary file path\n"
            "\t -h                help\n"
            , bin_name);
}

int main (int argc, char *argv[])
{
    int opt = 0;
    const char *bin_name = argv[0];
    char *source_file = NULL, *binary_file = NULL;
    size_t source_sizes = 0;
    size_t binary_sizes = 0;
    char *kernel_name = NULL;
    char *kernel_body = NULL;
    uint8_t *program_binaries = NULL;
    TestFileHandle source_fp, binary_fp;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    while ((opt = getopt (argc, argv, "i:o:h")) != -1) {
        switch (opt) {
        case 'i':
            source_file = optarg;
            break;
        case 'o':
            binary_file = optarg;
            break;
        case 'h':
            print_help (bin_name);
            return 0;

        default:
            print_help (bin_name);
            return -1;
        }
    }

    if (!source_file || !binary_file) {
        print_help (bin_name);
        return -1;
    }

    source_fp.fp = fopen (source_file, "r");
    binary_fp.fp = fopen (binary_file, "wb");
    if (!source_fp.fp || !binary_fp.fp) {
        XCAM_LOG_ERROR ("open source/binary file failed");
        return -1;
    }

    ret = get_source_sizes (source_fp, &source_sizes);
    CHECK (ret, "get source sizes from %s failed", source_file);

    kernel_body = (char *) xcam_malloc0 (sizeof (char) * (source_sizes + 1));
    XCAM_ASSERT(kernel_body);

    ret = read_source (source_fp, kernel_body, source_sizes);
    CHECK (ret, "read source from %s failed", source_file);
    kernel_body[source_sizes] = '\0';

    SmartPtr<CLContext> context;
    context = CLDevice::instance ()->get_context ();

    ret = get_kernel_name (source_file, &kernel_name);
    CHECK (ret, "get kernel name failed");

    SmartPtr<CLKernel> kernel = new CLKernel (context, kernel_name);
    xcam_free (kernel_name);

    kernel->load_from_source (kernel_body, strlen (kernel_body), &program_binaries, &binary_sizes);

    ret = write_binary (binary_fp, program_binaries, binary_sizes);
    CHECK (ret, "write binary to %s failed", binary_file);

    xcam_free (kernel_body);
    xcam_free (program_binaries);
    return 0;
}
