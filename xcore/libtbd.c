/*
** Copyright 2012-2013 Intel Corporation
**
** Licensed under the Apache License, Version 2.0 (the "License");
** you may not use this file except in compliance with the License.
** You may obtain a copy of the License at
**
**     http://www.apache.org/licenses/LICENSE-2.0
**
** Unless required by applicable law or agreed to in writing, software
** distributed under the License is distributed on an "AS IS" BASIS,
** WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
** See the License for the specific language governing permissions and
** limitations under the License.
*/

#include <stdbool.h> /* defines bool type */
#include <stddef.h>  /* defines size_t */
#include <stdint.h>  /* defines integer types with specified widths */
#include <stdio.h>   /* defines FILE */
#include <string.h>  /* defines memcpy and memset */

#include "libtbd.h"  /* our own header file */

/*!
* \brief Debug messages.
*/
#ifdef __ANDROID__
#define LOG_TAG "libtbd"
#include <utils/Log.h>
#define MSG_LOG(...) LOGD(__VA_ARGS__)
#define MSG_ERR(...) LOGE(__VA_ARGS__)
#else
#include <stdio.h>
#define MSG_LOG(...) fprintf(stdout, __VA_ARGS__); fprintf(stdout, "\n");
#define MSG_ERR(...) fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n");
#endif

/*
 * Checks the validity of the pointer
 * param[in]    a_ptr             Pointer to be examined
 * return                         True if pointer ok
 */
bool is_valid_pointer(void* a_ptr)
{
    if ((!a_ptr) || ((unsigned long)(a_ptr) % sizeof(uint32_t))) {
        return false;
    } else {
        return true;
    }
}

/*
 * Calculates checksum for a data block.
 * param[in]    a_data_ptr        Data from where to calculate the checksum
 * param[in]    a_data_size       Size of the data
 * return                         The checksum
 */
uint32_t get_checksum(void *a_data_ptr, size_t a_data_size)
{
    uint32_t *ptr32 = a_data_ptr;
    int size32 = a_data_size / sizeof(uint32_t);

    /* Simple checksum algorithm: summing up the data content
     * as 32-bit numbers */
    uint32_t checksum32 = 0;
    if (size32) {
        if (size32 & 0x01) {
            checksum32 += *ptr32++;
            size32 -= 1;
        }
        if (size32 & 0x02) {
            checksum32 += *ptr32++;
            checksum32 += *ptr32++;
            size32 -= 2;
        }
        for (; size32 > 0; size32 -= 4) {
            checksum32 += *ptr32++;
            checksum32 += *ptr32++;
            checksum32 += *ptr32++;
            checksum32 += *ptr32++;
        }
    }

    return checksum32;
}

/*
 * Common subroutine to validate Tagged Binary Data container, without
 * paying attention to checksum or data tagging. This function assumes
 * that the data resides in "legal" memory area as there is no size
 * given together with input pointer.
 * param[in]    a_data_ptr        Pointer to container
 * return                         Return code indicating possible errors
 */
tbd_error_t validate_anysize(void *a_data_ptr)
{
    uint8_t *byte_ptr, *eof_ptr;
    tbd_record_header_t *record_ptr;
    uint32_t record_size;

    /* Container should begin with a header */
    tbd_header_t *header_ptr = a_data_ptr;

    /* Check against illegal pointers */
    if (!is_valid_pointer(header_ptr)) {
        MSG_ERR("LIBTBD ERROR: Cannot access data!");
        return tbd_err_data;
    }

    /* Check that the indicated data size makes sense,
     * and is not too much or too little */
    if (header_ptr->size % sizeof(uint32_t)) {
        MSG_ERR("LIBTBD ERROR: Size in header should be multiple of 4 bytes!");
        return tbd_err_data;
    }
    if (header_ptr->size < sizeof(tbd_header_t)) {
        MSG_ERR("LIBTBD ERROR: Invalid data header!");
        return tbd_err_data;
    }

    /* First record is just after header, a byte pointer is needed
     * to do math with sizes and pointers */
    byte_ptr = (uint8_t *)(header_ptr + 1);
    eof_ptr = (uint8_t *)(a_data_ptr) + header_ptr->size;

    /* Loop until there are no more records to go */
    while (byte_ptr < eof_ptr) {
        /* At least one more record is expected */

        /* Record header must be within the given data size */
        if (byte_ptr + sizeof(tbd_record_header_t) > eof_ptr) {
            MSG_ERR("LIBTBD ERROR: Invalid data header!");
            return tbd_err_data;
        }

        record_ptr = (tbd_record_header_t *)(byte_ptr);
        record_size = record_ptr->size;

        /* Check that the indicated record size makes sense,
         * and is not too much or too little */
        if (record_size % sizeof(uint32_t)) {
            MSG_ERR("LIBTBD ERROR: Size in record should be multiple of 4 bytes!");
            return tbd_err_data;
        }
        if (record_size < sizeof(tbd_record_header_t)) {
            MSG_ERR("LIBTBD ERROR: Invalid record header!");
            return tbd_err_data;
        }
        if (byte_ptr + record_size > eof_ptr) {
            MSG_ERR("LIBTBD ERROR: Invalid record header!");
            return tbd_err_data;
        }

        /* This record ok, continue the while loop... */
        byte_ptr += record_size;
    }

    /* Seems that we have a valid data with no more headers */
    return tbd_err_none;
}

/*
 * Common subroutine to validate Tagged Binary Data, without paying
 * attention to checksum or data tagging. Also, this function does
 * check that the data fits in the given buffer size.
 * param[in]    a_data_ptr        Pointer to data buffer
 * param[in]    a_data_size       Size of the data buffer
 * return                         Return code indicating possible errors
 */
tbd_error_t validate(void *a_data_ptr, size_t a_data_size)
{
    /* Container should begin with a header */
    tbd_header_t *header_ptr = a_data_ptr;

    /* Check against illegal pointers */
    if (!is_valid_pointer(header_ptr)) {
        MSG_ERR("LIBTBD ERROR: Cannot access data!");
        return tbd_err_data;
    }

    /* Check that the TBD header fits into given data */
    if (sizeof(tbd_header_t) > a_data_size) {
        MSG_ERR("TBD ERROR: #1 Too small data buffer given!");
        return tbd_err_data;
    }

    /* Check that the indicated data fits in the buffer */
    if (header_ptr->size > a_data_size) {
        MSG_ERR("TBD ERROR: #2 Too small data buffer given!");
        return tbd_err_data;
    }

    /* Check the the content is ok */
    return validate_anysize(a_data_ptr);
}

/*
 * Creates a new, empty Tagged Binary Data container with the tag
 * that was given. Also updates the checksum and size accordingly.
 * Note that the buffer size must be large enough for the header
 * to fit in, the exact amount being 24 bytes (for tbd_header_t).
 * param[in]    a_data_ptr        Pointer to modifiable container buffer
 * param[in]    a_data_size       Size of the container buffer
 * param[in]    a_tag             Tag the container shall have
 * param[out]   a_new_size        Updated container size
 * return                         Return code indicating possible errors
 */
tbd_error_t tbd_create(void *a_data_ptr, size_t a_data_size
                       , tbd_tag_t a_tag, size_t *a_new_size)
{
    tbd_header_t *header_ptr;

    /* Check that the TBD header fits into given data */
    if (sizeof(tbd_header_t) > a_data_size) {
        MSG_ERR("LIBTBD ERROR: Not enough data given!");
        return tbd_err_argument;
    }

    /* Nullify everything */
    memset(a_data_ptr, 0, sizeof(tbd_header_t));

    /* The header is what we need */
    header_ptr = a_data_ptr;

    header_ptr->tag = a_tag;

    header_ptr->size = sizeof(tbd_header_t);
    header_ptr->version = IA_TBD_VERSION;
    header_ptr->revision = IA_TBD_REVISION;
    header_ptr->config_bits = 0;
    header_ptr->checksum = get_checksum(header_ptr, sizeof(tbd_header_t));

    *a_new_size = sizeof(tbd_header_t);

    return tbd_err_none;
}

/*
 * Performs number of checks to given Tagged Binary Data container,
 * including the verification of the checksum. The function does not
 * care about the tag type of the container.
 * param[in]    a_data_ptr        Pointer to container buffer
 * param[in]    a_data_size       Size of the container buffer
 * return                         Return code indicating possible errors
 */
tbd_error_t tbd_validate_anytag(void *a_data_ptr, size_t a_data_size)
{
    tbd_header_t *header_ptr;

    /* Check the the content is ok */
    int r;
    if ((r = validate(a_data_ptr, a_data_size))) {
        return r;
    }

    /* Container should begin with a header */
    header_ptr = a_data_ptr;

    /* Check that the checksum is correct */

    /* When calculating the checksum for the original data, the checksum
     * field has been filled with zero value - so after inserting the
     * checksum in its place, the new calculated checksum is actually
     * two times the original */

    if (get_checksum(header_ptr, header_ptr->size) - header_ptr->checksum != header_ptr->checksum) {
        MSG_ERR("LIBTBD ERROR: Checksum doesn't match!");
        return tbd_err_data;
    }

    /* Seems that we have valid data */
    return tbd_err_none;
}

/*
 * Performs number of checks to given Tagged Binary Data container,
 * including the verification of the checksum. Also, the data must have
 * been tagged properly. The tag is further used to check endianness,
 * and if it seems wrong, a specific debug message is printed out.
 * param[in]    a_data_ptr        Pointer to container buffer
 * param[in]    a_data_size       Size of the container buffer
 * param[in]    a_tag             Tag the data must have
 * return                         Return code indicating possible errors
 */
tbd_error_t tbd_validate(void *a_data_ptr, size_t a_data_size
                         , tbd_tag_t a_tag)
{
    tbd_header_t *header_ptr;

    /* Check the the content is ok */
    int r;
    if ((r = validate(a_data_ptr, a_data_size))) {
        return r;
    }

    /* Container should begin with a header */
    header_ptr = a_data_ptr;

    /* Check that the tag is correct */
    if (header_ptr->tag != a_tag) {
        /* See if we have wrong endianness or incorrect tag */
        uint32_t reverse_tag = ( (((a_tag) >> 24) & 0x000000FF)
                                 | (((a_tag) >> 8) & 0x0000FF00)
                                 | (((a_tag) << 8) & 0x00FF0000)
                                 | (((a_tag) << 24) & 0xFF000000) );

        if (reverse_tag == header_ptr->tag) {
            MSG_ERR("LIBTBD ERROR: Wrong endianness of data!");
        } else {
            MSG_ERR("LIBTBD ERROR: Data is not tagged properly!");
        }
        return tbd_err_data;
    }

    /* Check that the checksum is correct */

    /* When calculating the checksum for the original data, the checksum
     * field has been filled with zero value - so after inserting the
     * checksum in its place, the new calculated checksum is actually
     * two times the original */

    if (get_checksum(header_ptr, header_ptr->size) - header_ptr->checksum != header_ptr->checksum) {
        MSG_ERR("LIBTBD ERROR: Checksum doesn't match!");
        return tbd_err_data;
    }

    /* Seems that we have valid data */
    return tbd_err_none;
}

/*
 * Checks if a given kind of record exists in the Tagged Binary Data,
 * and if yes, tells the location of such record as well as its size.
 * If there are multiple records that match the query, the indicated
 * record is the first one.
 * param[in]    a_data_ptr        Pointer to container buffer
 * param[in]    a_record_class    Class the record must have
 * param[in]    a_record_format   Format the record must have
 * param[out]   a_record_data     Record data (or NULL if not found)
 * param[out]   a_record_size     Record size (or 0 if not found)
 * return                         Return code indicating possible errors
 */
tbd_error_t tbd_get_record(void *a_data_ptr
                           , tbd_class_t a_record_class, tbd_format_t a_record_format
                           , void **a_record_data, uint32_t *a_record_size)
{
    tbd_header_t *header_ptr;
    uint8_t *byte_ptr, *eof_ptr;

    /* Check the the content is ok */
    int r;
    if ((r = validate_anysize(a_data_ptr))) {
        return r;
    }

    /* Container should begin with a header */
    header_ptr = a_data_ptr;

    /* First record is just after header */
    byte_ptr = (uint8_t *)(header_ptr + 1);
    eof_ptr = (uint8_t *)(a_data_ptr) + header_ptr->size;

    /* Loop until there are no more records to go */
    while (byte_ptr < eof_ptr) {
        /* At least one more record is expected */
        tbd_record_header_t *record_ptr = (tbd_record_header_t *)(byte_ptr);

        uint16_t record_class = record_ptr->class_id;
        uint8_t  record_format = record_ptr->format_id;
        uint32_t record_size = record_ptr->size;

        if (((a_record_class == tbd_class_any) || (a_record_class == record_class))
                && ((a_record_format == tbd_format_any) || (a_record_format == record_format))) {

            /* Match found */
            *a_record_data = record_ptr + 1;
            *a_record_size = record_size - sizeof(tbd_record_header_t);

            return tbd_err_none;

        }

        /* Match not found yet, continue the while loop... */
        byte_ptr += record_size;
    }

    MSG_LOG("libtbd: Record not found!");
    *a_record_data = NULL;
    *a_record_size = 0;
    return tbd_err_none;
}

/*
 * The given record is inserted into the Tagged Binary Data container
 * that must exist already. New records are always added to the end,
 * regardless if a record with the same class and format field already
 * exists in the data. Also updates the checksum and size accordingly.
 * Note that the buffer size must be large enough for the inserted
 * record to fit in, the exact amount being the size of original
 * Tagged Binary Data container plus the size of record data to be
 * inserted plus 8 bytes (for tbd_record_header_t).
 * param[in]    a_data_ptr        Pointer to modifiable container buffer
 * param[in]    a_data_size       Size of buffer (surplus included)
 * param[in]    a_record_class    Class the record shall have
 * param[in]    a_record_format   Format the record shall have
 * param[in]    a_record_data     Record data
 * param[in]    a_record_size     Record size
 * param[out]   a_new_size        Updated container size
 * return                         Return code indicating possible errors
 */
tbd_error_t tbd_insert_record(void *a_data_ptr, size_t a_data_size
                              , tbd_class_t a_record_class, tbd_format_t a_record_format
                              , void *a_record_data, size_t a_record_size
                              , size_t *a_new_size)
{
    tbd_header_t *header_ptr;
    size_t new_size;
    tbd_record_header_t *record_ptr;
    int r;

    /* Check the the content is ok */
    if ((r = validate(a_data_ptr, a_data_size))) {
        return r;
    }

    /* Container should begin with a header */
    header_ptr = a_data_ptr;

    /* Check that the new record fits into given data */
    new_size = header_ptr->size + sizeof(tbd_record_header_t) + a_record_size;

    if (new_size > a_data_size) {
        MSG_ERR("LIBTBD ERROR: #3 Too small data buffer given!");
        return tbd_err_argument;
    }

    /* Check against illegal pointers */
    if (!is_valid_pointer(a_record_data)) {
        MSG_ERR("LIBTBD ERROR: Cannot access data!");
        return tbd_err_data;
    }

    /* Check that the indicated data size makes sense */
    if (a_record_size % sizeof(uint32_t)) {
        MSG_ERR("LIBTBD ERROR: Size in record should be multiple of 4 bytes!");
        return tbd_err_data;
    }

    /* Where our record should go */
    record_ptr = (tbd_record_header_t *)((char *)(a_data_ptr) + header_ptr->size);

    /* Create record header and store the record itself */
    record_ptr->size = sizeof(tbd_record_header_t) + a_record_size;
    record_ptr->format_id   = a_record_format;
    record_ptr->packing_key = 0;
    record_ptr->class_id    = a_record_class;
    record_ptr++;
    memcpy(record_ptr, a_record_data, a_record_size);

    /* Update the header */
    header_ptr->size = new_size;
    header_ptr->checksum = 0;
    header_ptr->checksum = get_checksum(header_ptr, new_size);

    *a_new_size = new_size;

    return tbd_err_none;
}

/*
 * The indicated record is removed from the Tagged Binary Data, after
 * which the checksum and size are updated accordingly. If there are
 * multiple records that match the class and format, only the first
 * instance is removed. If no record is found, nothing will be done.
 * Note that the resulting Tagged Binary Data container will
 * be smaller than the original, but it does not harm to store the
 * resulting container in its original length, either.
 * param[in]    a_data_ptr        Pointer to modifiable container buffer
 * param[in]    a_record_class    Class the record should have
 * param[in]    a_record_format   Format the record should have
 * param[out]   a_new_size        Updated container size
 * return                         Return code indicating possible errors
 */
tbd_error_t tbd_remove_record(void *a_data_ptr
                              , tbd_class_t a_record_class, tbd_format_t a_record_format
                              , size_t *a_new_size)
{
    tbd_header_t *header_ptr;
    uint8_t *byte_ptr, *eof_ptr;
    size_t new_size;

    /* Check the the content is ok */
    int r;
    if ((r = validate_anysize(a_data_ptr))) {
        return r;
    }

    /* Container should begin with a header */
    header_ptr = a_data_ptr;

    /* First record is just after header */
    byte_ptr = (uint8_t *)(header_ptr + 1);
    eof_ptr = (uint8_t *)(a_data_ptr) + header_ptr->size;

    /* Loop until there are no more records to go */
    while (byte_ptr < eof_ptr) {
        /* At least one more record is expected */
        tbd_record_header_t *record_ptr = (tbd_record_header_t *)(byte_ptr);

        uint16_t record_class = record_ptr->class_id;
        uint8_t  record_format = record_ptr->format_id;
        uint32_t record_size = record_ptr->size;

        if (((a_record_class == tbd_class_any) || (a_record_class == record_class))
                && ((a_record_format == tbd_format_any) || (a_record_format == record_format))) {

            /* Match found, remove the record */
            memcpy(byte_ptr, byte_ptr + record_size, eof_ptr - (byte_ptr + record_size));

            /* Update the header */
            new_size = header_ptr->size - record_size;
            header_ptr->size = new_size;
            header_ptr->checksum = 0;
            header_ptr->checksum = get_checksum(header_ptr, new_size);

            *a_new_size = new_size;

            return tbd_err_none;

        }

        /* Match not found yet, continue the while loop... */
        byte_ptr += record_size;
    }

    MSG_LOG("libtbd: Record not found!");
    *a_new_size = header_ptr->size;
    return tbd_err_none;
}

/*
 * Validates the Tagged Binary data container and generates a human
 * readable detailed report on the content, including information about
 * the records contained.
 * param[in]    a_data_ptr        Pointer to container buffer
 * param[in]    a_data_size       Size of the container buffer
 * param[in]    a_outfile         Pointer to open file (may be stdout)
 * return                         Return code indicating possible errors
 */
tbd_error_t tbd_infoprint(void *a_data_ptr, size_t a_data_size
                          , FILE *a_outfile)
{
    tbd_header_t *header_ptr;
    uint8_t *byte_ptr, *eof_ptr, record_format, record_packing;
    int num_of_records = 0, total_data = 0;
    uint16_t record_class;
    uint32_t record_size;

    /* Check the the content is ok */
    int r;
    if ((r = validate(a_data_ptr, a_data_size))) {
        return r;
    }

    /* Container should begin with a header */
    header_ptr = a_data_ptr;

    fprintf(a_outfile, "Data tag:      0x%08x (\'%c\' \'%c\' \'%c\' \'%c\')\n", header_ptr->tag, ((char *)(&header_ptr->tag))[0], ((char *)(&header_ptr->tag))[1], ((char *)(&header_ptr->tag))[2], ((char *)(&header_ptr->tag))[3]);
    fprintf(a_outfile, "Data size:     %d (0x%x), buffer size %d (0x%x)\n", header_ptr->size, header_ptr->size, a_data_size, a_data_size);
    fprintf(a_outfile, "Data version:  0x%08x\n", header_ptr->version);
    fprintf(a_outfile, "Data revision: 0x%08x\n", header_ptr->revision);
    fprintf(a_outfile, "Data config:   0x%08x\n", header_ptr->config_bits);
    fprintf(a_outfile, "Data checksum: 0x%08x\n", header_ptr->checksum);

    fprintf(a_outfile, "\n");

    /* First record is just after header */
    byte_ptr = (uint8_t *)(header_ptr + 1);
    eof_ptr = (uint8_t *)(a_data_ptr) + header_ptr->size;

    /* Loop until there are no more records to go */
    while (byte_ptr < eof_ptr) {
        /* At least one more record is expected */
        tbd_record_header_t *record_ptr = (tbd_record_header_t *)(byte_ptr);
        num_of_records++;

        record_class = record_ptr->class_id;
        record_format = record_ptr->format_id;
        record_packing = record_ptr->packing_key;
        record_size = record_ptr->size;
        total_data += record_size - sizeof(tbd_record_header_t);

        fprintf(a_outfile, "Record size:     %d (0x%x)\n", record_size, record_size);
        fprintf(a_outfile, "Size w/o header: %d (0x%x)\n", record_size - sizeof(tbd_record_header_t), record_size - sizeof(tbd_record_header_t));
        fprintf(a_outfile, "Record class:    %d", record_class);
        switch (record_class) {
        case tbd_class_any:
            fprintf(a_outfile, " \"tbd_class_any\"\n");
            break;
        case tbd_class_aiq:
            fprintf(a_outfile, " \"tbd_class_aiq\"\n");
            break;
        case tbd_class_drv:
            fprintf(a_outfile, " \"tbd_class_drv\"\n");
            break;
        case tbd_class_hal:
            fprintf(a_outfile, " \"tbd_class_hal\"\n");
            break;
        default:
            fprintf(a_outfile, " (unknown class)\n");
            break;
        }
        fprintf(a_outfile, "Record format:   %d", record_format);
        switch (record_format) {
        case tbd_format_any:
            fprintf(a_outfile, " \"tbd_format_any\"\n");
            break;
        case tbd_format_custom:
            fprintf(a_outfile, " \"tbd_format_custom\"\n");
            break;
        case tbd_format_container:
            fprintf(a_outfile, " \"tbd_format_container\"\n");
            break;
        default:
            fprintf(a_outfile, " (unknown format)\n");
            break;
        }
        fprintf(a_outfile, "Packing:         %d", record_packing);
        if (record_packing == 0) {
            fprintf(a_outfile, " (no packing)\n");
        } else {
            fprintf(a_outfile, "\n");
        }

        fprintf(a_outfile, "\n");

        /* Continue the while loop... */
        byte_ptr += record_size;
    }

    fprintf(a_outfile, "Number of records found: %d\n", num_of_records);
    fprintf(a_outfile, "Total data in records: %d bytes (without headers)\n", total_data);
    fprintf(a_outfile, "\n");
    return tbd_err_none;
}

