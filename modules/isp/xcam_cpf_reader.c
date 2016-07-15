/*
 * xcam_thread.h - xcam basic thread
 *
 *  Copyright (c) 2014 Intel Corporation
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


#include "xcam_cpf_reader.h"

#include <string.h>
#include <stdlib.h>
#include "libtbd.h"

#undef XCAM_FAIL_RETURN_VAL
#define XCAM_FAIL_RETURN_VAL(exp, ret) \
    if (!(exp)) {                  \
        XCAM_LOG_WARNING ("XCAM_FAIL_RETURN_VAL %s", #exp);  \
        return ret;                \
    }

#undef XCAM_FAIL_RETURN
#define XCAM_FAIL_RETURN(exp) \
    if (!(exp)) {                  \
        XCAM_LOG_WARNING ("XCAM_FAIL_RETURN %s", #exp);  \
        return ;                \
    }

void *xcam_new0(size_t size)
{
    void *buf = malloc (size);
    memset (buf, 0, size);
    return buf;
}

XCamCpfBlob * xcam_cpf_blob_new ()
{
    return (XCamCpfBlob*) xcam_new0 (sizeof(XCamCpfBlob));
}

void xcam_cpf_blob_free (XCamCpfBlob *blob)
{
    XCAM_FAIL_RETURN (blob);

    if (blob->data)
        xcam_free (blob->data);

    xcam_free (blob);
}

static int32_t
read_cpf_file (const char *cpf_file, uint8_t **buf)
{
    int32_t size = 0;
    FILE *fp = fopen (cpf_file, "rb");
    XCAM_FAIL_RETURN_VAL (fp, -1);

    *buf = NULL;

    if (fseek (fp, 0, SEEK_END) < 0)
        goto read_error;
    if ((size = ftell (fp)) <= 0)
        goto read_error;
    if (fseek( fp, 0, SEEK_SET) < 0)
        goto read_error;

    *buf = (uint8_t*) xcam_new0 (size);
    XCAM_ASSERT (*buf);
    if (fread (*buf, 1, size, fp) != (size_t) size)
        goto read_error;

    fclose (fp);
    return size;

read_error:
    XCAM_LOG_ERROR ("read cpf(%s) failed", cpf_file);
    fclose (fp);
    if (*buf) {
        xcam_free (*buf);
        *buf = NULL;
    }
    return -1;

}

boolean
xcam_cpf_read (const char *cpf_file, XCamCpfBlob *aiq_cpf, XCamCpfBlob *hal_cpf)
{
    uint8_t *cpf_buf;
    int32_t cpf_size;

    uint8_t *blob;
    uint32_t blob_size;

    XCAM_FAIL_RETURN_VAL (cpf_file, FALSE);
    XCAM_FAIL_RETURN_VAL (aiq_cpf, FALSE);

    /* read cpf */
    if ((cpf_size = read_cpf_file(cpf_file, &cpf_buf)) <= 0) {
        XCAM_LOG_ERROR ("read cpf_file(%s) failed.", cpf_file);
        return FALSE;
    }

    /* check sum */
    if (tbd_validate (cpf_buf, cpf_size, tbd_tag_cpff) != tbd_err_none) {
        XCAM_LOG_ERROR ("tbd validate cpf file(%s) failed.", cpf_file);
        goto free_buf;
    }

    /* fetch AIQ */
    if ( (tbd_get_record (cpf_buf, tbd_class_aiq, tbd_format_any,
                          (void**)&blob, &blob_size) != tbd_err_none) ||
            !blob || blob_size <= 0) {
        XCAM_LOG_ERROR ("CPF parse AIQ record failed.");
        goto free_buf;
    }
    aiq_cpf->data = (uint8_t*) malloc (blob_size);
    XCAM_ASSERT (aiq_cpf->data);
    aiq_cpf->size = blob_size;
    memcpy (aiq_cpf->data, blob, blob_size);


#if 0 //DRV not necessary
    /* fetch DRV */
    if (tbd_get_record (cpf_buf, tbd_class_drv, tbd_format_any,
                        &drv_blob.data, &drv_blob.size) != tbd_err_none) {
        XCAM_LOG_ERROR ("CPF parse DRV record failed.");
        return FALSE;
    }
#endif


    /* fetch HAL */
    if (hal_cpf) {
        if (tbd_get_record (cpf_buf, tbd_class_hal, tbd_format_any,
                            (void**)&blob, &blob_size) != tbd_err_none) {
            XCAM_LOG_WARNING ("CPF doesn't have HAL record.");
            // ignore HAL, not necessary
        } else if (blob && blob_size > 0) {
            hal_cpf->data = (uint8_t*) malloc (blob_size);
            XCAM_ASSERT (hal_cpf->data);
            hal_cpf->size = blob_size;
            memcpy (hal_cpf->data, blob, blob_size);
        }
    }

    xcam_free (cpf_buf);
    return TRUE;

free_buf:
    xcam_free (cpf_buf);
    return FALSE;

}
