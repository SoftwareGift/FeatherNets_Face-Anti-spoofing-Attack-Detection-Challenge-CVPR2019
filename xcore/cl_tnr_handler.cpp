/*
 * cl_tnr_handler.cpp - CL tnr handler
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
 * Author: Wei Zong <wei.zong@intel.com>
 */
#include "xcam_utils.h"
#include "x3a_stats_pool.h"
#include "cl_tnr_handler.h"

namespace XCam {

CLTnrImageKernel::CLTnrHistogram::CLTnrHistogram() {
    hor_hist_bin = 0;
    ver_hist_bin = 0;
    hor_hist_current = NULL;
    hor_hist_reference = NULL;
    ver_hist_current = NULL;
    ver_hist_reference = NULL;
};

CLTnrImageKernel::CLTnrHistogram::CLTnrHistogram(uint32_t width, uint32_t height) {
    hor_hist_bin = width;
    ver_hist_bin = height;
    if ((NULL == hor_hist_current) && (hor_hist_bin != 0)) {
        hor_hist_current = (float*)xcam_malloc0(hor_hist_bin * sizeof(float));
    }
    if ((NULL == ver_hist_current) && (ver_hist_bin != 0)) {
        ver_hist_current = (float*)xcam_malloc0(ver_hist_bin * sizeof(float));
    }
    if ((NULL == hor_hist_reference) && (hor_hist_bin != 0)) {
        hor_hist_reference = (float*)xcam_malloc0(hor_hist_bin * sizeof(float));
    }
    if ((NULL == ver_hist_reference) && (ver_hist_bin != 0)) {
        ver_hist_reference = (float*)xcam_malloc0(ver_hist_bin * sizeof(float));
    }
};

CLTnrImageKernel::CLTnrHistogram::~CLTnrHistogram() {
    if (NULL != hor_hist_current) {
        xcam_free(hor_hist_current);
        hor_hist_current = NULL;
    }
    if (NULL != ver_hist_current) {
        xcam_free(ver_hist_current);
        ver_hist_current = NULL;
    }
    if (NULL != hor_hist_reference) {
        xcam_free(hor_hist_reference);
        hor_hist_reference = NULL;
    }
    if (NULL != ver_hist_reference) {
        xcam_free(ver_hist_reference);
        ver_hist_reference = NULL;
    }
    hor_hist_bin = 0;
    ver_hist_bin = 0;
}

CLTnrImageKernel::CLTnrImageKernel (SmartPtr<CLContext> &context,
                                    const char *name,
                                    CLTnrType type)
    : CLImageKernel (context, name, false)
    , _type (type)
    , _gain_yuv (1.0)
    , _thr_y (0.05)
    , _thr_uv (0.05)
    , _gain_rgb (0.0)
    , _thr_r (0.064)  // set high initial threshold to get strong denoise effect
    , _thr_g (0.045)
    , _thr_b (0.073)
    , _frame_count (TNR_PROCESSING_FRAME_COUNT)
    , _stable_frame_count (1)
{
}

XCamReturn
CLTnrImageKernel::prepare_arguments (
    SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
    CLArgument args[], uint32_t &arg_count,
    CLWorkSize &work_size)
{
    SmartPtr<CLContext> context = get_context ();

    const VideoBufferInfo & video_info = input->get_video_info ();
    memset(_motion_info, 0, TNR_GRID_HOR_COUNT * TNR_GRID_VER_COUNT * sizeof(CLTnrMotionInfo));

    _image_in = new CLVaImage (context, input);
    if (CL_TNR_TYPE_RGB == _type) {
        // analyze motion between the latest adjacent two frames
        // Todo: enable analyze when utilize motion compensation next step
#if 0
        _gain = analyze_motion (input, CL_TNR_ANALYZE_STATS, _motion_info);

        if (_gain < 0.001) {
            XCAM_LOG_DEBUG ("_gain(%f) reset stable frame count!", _gain);
            _stable_frame_count = 1 ;
        } else {
            XCAM_LOG_DEBUG ("_gain(%f) increase stable frame count(%d)", _gain, _stable_frame_count);
            _stable_frame_count ++ ;
        }

        if (_stable_frame_count < TNR_PROCESSING_FRAME_COUNT) {
            _gain = 0.0;
        } else {
            _stable_frame_count = TNR_PROCESSING_FRAME_COUNT;
        }
#endif
        if (_image_in_list.size () < TNR_LIST_FRAME_COUNT) {
            while (_image_in_list.size () < TNR_LIST_FRAME_COUNT) {
                _image_in_list.push_back (_image_in);
            }
        } else {
            _image_in_list.pop_front ();
            _image_in_list.push_back (_image_in);
        }
    }

    _image_out = new CLVaImage (context, output);

    if (CL_TNR_TYPE_YUV == _type) {
        if (!_image_out_prev.ptr ()) {
            _image_out_prev = _image_in;
        }
    }
    _vertical_offset = video_info.aligned_height;

    XCAM_ASSERT (_image_in->is_valid () && _image_out->is_valid ());
    XCAM_FAIL_RETURN (
        WARNING,
        _image_in->is_valid () && _image_out->is_valid (),
        XCAM_RETURN_ERROR_MEM,
        "cl image kernel(%s) in/out memory not available", get_kernel_name ());

    //set args;
    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    work_size.local[0] = 8;
    work_size.local[1] = 4;
    if (CL_TNR_TYPE_YUV == _type) {
        args[0].arg_adress = &_image_in->get_mem_id ();
        args[0].arg_size = sizeof (cl_mem);

        args[1].arg_adress = &_image_out_prev->get_mem_id ();
        args[1].arg_size = sizeof (cl_mem);

        args[2].arg_adress = &_image_out->get_mem_id ();
        args[2].arg_size = sizeof (cl_mem);

        args[3].arg_adress = &_vertical_offset;
        args[3].arg_size = sizeof (_vertical_offset);

        args[4].arg_adress = &_gain_yuv;
        args[4].arg_size = sizeof (_gain_yuv);

        args[5].arg_adress = &_thr_y;
        args[5].arg_size = sizeof (_thr_y);

        args[6].arg_adress = &_thr_uv;
        args[6].arg_size = sizeof (_thr_uv);

        work_size.global[0] = video_info.width / 2;
        work_size.global[1] = video_info.height / 2;
        arg_count = 7;
    }
    else if (CL_TNR_TYPE_RGB == _type) {
        const CLImageDesc out_info = _image_out->get_image_desc ();
        work_size.global[0] = out_info.width;
        work_size.global[1] = out_info.height;

        args[0].arg_adress = &_image_out->get_mem_id ();
        args[0].arg_size = sizeof (cl_mem);

        args[1].arg_adress = &_gain_rgb;
        args[1].arg_size = sizeof (_gain_rgb);

        args[2].arg_adress = &_thr_r;
        args[2].arg_size = sizeof (_thr_r);

        args[3].arg_adress = &_thr_g;
        args[3].arg_size = sizeof (_thr_g);

        args[4].arg_adress = &_thr_b;
        args[4].arg_size = sizeof (_thr_b);

        args[5].arg_adress = &_frame_count;
        args[5].arg_size = sizeof (_frame_count);

        uint8_t index = 0;
        for (std::list<SmartPtr<CLImage>>::iterator it = _image_in_list.begin (); it != _image_in_list.end (); it++) {
            args[6 + index].arg_adress = &(*it)->get_mem_id ();
            args[6 + index].arg_size = sizeof (cl_mem);
            index++;
        }

        arg_count = 6 + index;
    }

    return XCAM_RETURN_NO_ERROR;
}


XCamReturn
CLTnrImageKernel::post_execute ()
{
    if ((CL_TNR_TYPE_YUV == _type) && _image_out->is_valid ()) {
        _image_out_prev = _image_out;
    }

    return CLImageKernel::post_execute ();
}

bool
CLTnrImageKernel::set_framecount (uint8_t count)
{
    // frame count only support 2/3/4/.
    XCAM_ASSERT (count >= 2 && count <= 4);

    _frame_count = count;
    return true;
}

bool
CLTnrImageKernel::set_rgb_config (const XCam3aResultTemporalNoiseReduction& config)
{
    _gain_rgb = (float)config.gain;
    _thr_r = (float)config.threshold[0];
    _thr_g = (float)config.threshold[1];
    _thr_b = (float)config.threshold[2];
    XCAM_LOG_DEBUG ("set TNR RGB config: _gain(%f), _thr_r(%f), _thr_g(%f), _thr_b(%f)",
                    _gain_rgb, _thr_r, _thr_g, _thr_b);

    return true;
}

bool
CLTnrImageKernel::set_yuv_config (const XCam3aResultTemporalNoiseReduction& config)
{
    _gain_yuv = (float)config.gain;
    _thr_y = (float)config.threshold[0];
    _thr_uv = (float)config.threshold[1];
    XCAM_LOG_DEBUG ("set TNR YUV config: _gain(%f), _thr_y(%f), _thr_uv(%f)",
                    _gain_yuv, _thr_y, _thr_uv);

    return true;
}

float
CLTnrImageKernel::analyze_motion (SmartPtr<DrmBoBuffer>& input, CLTnrAnalyzeDateType type, CLTnrMotionInfo* info)
{
    bool ret = true;

    SmartPtr<X3aStats> stats = NULL;
    XCam3AStats *stats_ptr = NULL;

    uint32_t hor_hist_bin = 0;
    uint32_t ver_hist_bin = 0;

    CLTnrMotionInfo motion_info;
    uint32_t sum_hor_shift = 0;
    uint32_t sum_ver_shift = 0;
    int32_t max_hor_shift = 0;
    int32_t max_ver_shift = 0;
    uint32_t unrelated_block_count = 0;
    uint32_t precise_factor = 1;

    float tnr_gain = 1.0;

    if (CL_TNR_ANALYZE_STATS == type) {
        stats = input->find_3a_stats ();
        if (!stats.ptr ()) {
            return 0;
        }
        stats_ptr = stats->get_stats ();
        hor_hist_bin = stats_ptr->info.width;
        ver_hist_bin = stats_ptr->info.height;
    } else {
        hor_hist_bin = input->get_video_info ().width;
        ver_hist_bin = input->get_video_info ().height;
    }

    // initialize image horizontal & vertical histogram
    if ((_image_histogram.hor_hist_bin == 0) && (_image_histogram.ver_hist_bin == 0)) {
        _image_histogram.hor_hist_bin = hor_hist_bin;
        _image_histogram.ver_hist_bin = ver_hist_bin;
        _image_histogram.hor_hist_current = (float*)xcam_malloc0(_image_histogram.hor_hist_bin * sizeof(float));
        _image_histogram.ver_hist_current = (float*)xcam_malloc0(_image_histogram.ver_hist_bin * sizeof(float));
        _image_histogram.hor_hist_reference = (float*)xcam_malloc0(_image_histogram.hor_hist_bin * sizeof(float));
        _image_histogram.ver_hist_reference = (float*)xcam_malloc0(_image_histogram.ver_hist_bin * sizeof(float));
    }

    if (CL_TNR_ANALYZE_STATS == type) {
        ret &= calculate_image_histogram(stats_ptr, CL_TNR_HIST_HOR_PROJECTION, _image_histogram.hor_hist_current);
        ret &= calculate_image_histogram(stats_ptr, CL_TNR_HIST_VER_PROJECTION, _image_histogram.ver_hist_current);
        precise_factor = 1;
    } else if (CL_TNR_ANALYZE_RGB == type) {
        ret &= calculate_image_histogram(input, CL_TNR_HIST_HOR_PROJECTION, _image_histogram.hor_hist_current);
        ret &= calculate_image_histogram(input, CL_TNR_HIST_VER_PROJECTION, _image_histogram.ver_hist_current);
        precise_factor = 2;
    } else {
        return 0;
    }

    if (ret &&
            (NULL != _image_histogram.hor_hist_current) &&
            (NULL != _image_histogram.hor_hist_reference) &&
            (NULL != _image_histogram.ver_hist_current) &&
            (NULL != _image_histogram.ver_hist_reference)) {

        uint32_t grid_width = hor_hist_bin / TNR_GRID_HOR_COUNT;
        uint32_t grid_height = ver_hist_bin / TNR_GRID_VER_COUNT;

        //print_image_histogram ();

        for (uint8_t index_hor = 0; index_hor < TNR_GRID_HOR_COUNT; index_hor++) {
            for (uint8_t index_ver = 0; index_ver < TNR_GRID_VER_COUNT; index_ver++) {
                ret &= detect_motion(_image_histogram.hor_hist_current + index_hor * grid_width, _image_histogram.hor_hist_reference + index_hor * grid_width, grid_width, motion_info.hor_shift, motion_info.hor_corr);
                //XCAM_LOG_DEBUG ("motion_info  hor corr = %f, ver corr = %f", motion_info.hor_corr, motion_info.ver_corr);
                ret &= detect_motion(_image_histogram.ver_hist_current + index_ver * grid_height, _image_histogram.ver_hist_reference + index_ver * grid_height, grid_height, motion_info.ver_shift, motion_info.ver_corr);
                //XCAM_LOG_DEBUG ("motion_info  hor shift = %d, ver shift = %d", motion_info.hor_shift, motion_info.ver_shift);
                info[index_hor * TNR_GRID_HOR_COUNT + index_ver].hor_shift = motion_info.hor_shift;
                info[index_hor * TNR_GRID_HOR_COUNT + index_ver].ver_shift = motion_info.ver_shift;
                info[index_hor * TNR_GRID_HOR_COUNT + index_ver].hor_corr = motion_info.hor_corr;
                info[index_hor * TNR_GRID_HOR_COUNT + index_ver].ver_corr = motion_info.ver_corr;

                if (motion_info.hor_corr > 0.5) {
                    sum_hor_shift += abs(motion_info.hor_shift);
                    if (abs(max_hor_shift) < abs(motion_info.hor_shift)) {
                        max_hor_shift = motion_info.hor_shift;
                    }
                } else {
                    unrelated_block_count ++;
                }
                if (motion_info.ver_corr > 0.5) {
                    sum_ver_shift += abs(motion_info.ver_shift);
                    if (abs(max_ver_shift) < abs(motion_info.ver_shift)) {
                        max_ver_shift = motion_info.ver_shift;
                    }
                } else {
                    unrelated_block_count ++;
                }
            }
        }
    } else {
        return 0;
    }

    if (ret) {
        // copy current histogram to reference and "erase" current histogram content, waiting for next frame calculation
        memcpy(_image_histogram.hor_hist_reference, _image_histogram.hor_hist_current, hor_hist_bin * sizeof(float));
        memcpy(_image_histogram.ver_hist_reference, _image_histogram.ver_hist_current, ver_hist_bin * sizeof(float));
        memset(_image_histogram.hor_hist_current, 0, hor_hist_bin * sizeof(float));
        memset(_image_histogram.ver_hist_current, 0, ver_hist_bin * sizeof(float));
    }

    XCAM_LOG_DEBUG ("max_hor_shift = %d, max_ver_shift = %d, sum_hor_shift = %d, sum_ver_shift = %d, unrelated_block_count = %d ===",
                    max_hor_shift, max_ver_shift, sum_hor_shift, sum_ver_shift, unrelated_block_count);
    // the motion threshold should be higher when calculate with lager resolution image or stats data
    if ( (abs(max_hor_shift) >= abs(TNR_MOTION_THRESHOLD * precise_factor)) || (abs(max_ver_shift) >= abs(TNR_MOTION_THRESHOLD * precise_factor)) ||
            (sum_hor_shift >= TNR_GRID_HOR_COUNT) || (sum_ver_shift >= TNR_GRID_VER_COUNT) ||
            (unrelated_block_count >= (TNR_GRID_HOR_COUNT >> 1)) ) {
        // TODO: assign value "0 ~ 1" to tnr_gain according to motion information
        tnr_gain = 0;
        XCAM_LOG_DEBUG("Motion detect SKIP TNR !!! ===");
    }

    return tnr_gain;
}

bool
CLTnrImageKernel::calculate_image_histogram (XCam3AStats* stats, CLTnrHistogramType type, float* histogram)
{
    if ( NULL == stats || NULL == histogram ) {
        return false;
    }

    uint32_t normalize_factor = (1 << stats->info.bit_depth) - 1;
    uint32_t image_width = stats->info.width;
    uint32_t image_height = stats->info.height;
    uint32_t image_aligned_width = stats->info.aligned_width;
    uint32_t hor_hist_bin = image_width;
    uint32_t ver_hist_bin = image_height;

    switch (type) {
    case CL_TNR_HIST_HOR_PROJECTION :
        for (uint32_t bin = 0; bin < hor_hist_bin; bin++) {
            for (uint32_t row_index = 0; row_index < image_height; row_index++) {
                histogram[bin] += (float)(stats->stats[row_index * image_aligned_width + bin].avg_y)
                                  / (1.0 * normalize_factor);
            }
        }
        break;
    case CL_TNR_HIST_VER_PROJECTION :
        for (uint32_t bin = 0; bin < ver_hist_bin; bin++) {
            for (uint32_t col_index = 0; col_index < image_width; col_index++) {
                histogram[bin] += (float)(stats->stats[col_index + bin * image_aligned_width].avg_y)
                                  / (1.0 * normalize_factor);
            }
        }
        break;
    case CL_TNR_HIST_BRIGHTNESS :
        for (uint32_t row_index = 0; row_index < image_height; row_index++) {
            for (uint32_t col_index = 0; col_index < image_width; col_index++) {
                uint8_t bin = (stats->stats[row_index * image_aligned_width + col_index].avg_y * 255)
                              / normalize_factor;
                histogram[bin]++;
            }
        }
        break;
    default :
        break;
    }

    return true;
}

bool
CLTnrImageKernel::calculate_image_histogram (SmartPtr<DrmBoBuffer> &input, CLTnrHistogramType type, float* histogram)
{
    if ( NULL == histogram ) {
        return false;
    }

    uint32_t normalize_factor = (1 << input->get_video_info ().color_bits) - 1;
    uint32_t image_width = input->get_video_info ().width;
    uint32_t image_height = input->get_video_info ().height;
    uint32_t image_aligned_width = input->get_video_info ().aligned_width;
    uint32_t stride = input->get_video_info ().strides[0];

    uint32_t hor_hist_bin = image_width;
    uint32_t ver_hist_bin = image_height;
    uint32_t pxiel_bytes = stride / image_aligned_width;

    uint32_t format = input->get_video_info ().format;
    if (XCAM_PIX_FMT_RGBA64 != format) {
        XCAM_LOG_ERROR ("Only support RGBA64 format !");
        return false;
    }

    uint8_t* image_buffer = input->map();
    if (NULL == image_buffer) {
        return false;
    }

    switch (type) {
    case CL_TNR_HIST_HOR_PROJECTION :
        for (uint32_t bin = 0; bin < hor_hist_bin; bin++) {
            for (uint32_t row_index = 0; row_index < image_height; row_index++) {
                histogram[bin] += (float)(image_buffer[row_index * stride + pxiel_bytes * bin] +
                                          (image_buffer[row_index * stride + pxiel_bytes * bin + 1] << 8) +
                                          image_buffer[row_index * stride + pxiel_bytes * bin + 2] +
                                          (image_buffer[row_index * stride + pxiel_bytes * bin + 3] << 8) +
                                          image_buffer[row_index * stride + pxiel_bytes * bin + 4] +
                                          (image_buffer[row_index * stride + pxiel_bytes * bin + 5] << 8) )
                                  / (3.0 * normalize_factor);
            }
        }
        break;
    case CL_TNR_HIST_VER_PROJECTION :
        for (uint32_t bin = 0; bin < ver_hist_bin; bin++) {
            for (uint32_t col_index = 0; col_index < stride; col_index += pxiel_bytes) {
                histogram[bin] += (float)(image_buffer[col_index + bin * stride] +
                                          (image_buffer[col_index + bin * stride + 1] << 8) +
                                          image_buffer[col_index + bin * stride + 2] +
                                          (image_buffer[col_index + bin * stride + 3] << 8) +
                                          image_buffer[col_index + bin * stride + 4] +
                                          (image_buffer[col_index + bin * stride + 5] << 8) )
                                  / (3.0 * normalize_factor);
            }
        }
        break;
    case CL_TNR_HIST_BRIGHTNESS :
        for (uint32_t row_index = 0; row_index < image_height; row_index++) {
            for (uint32_t col_index = 0; col_index < stride; col_index += pxiel_bytes) {
                uint8_t bin = (image_buffer[row_index * stride + col_index] +
                               (image_buffer[row_index * stride + col_index + 1] << 8) +
                               image_buffer[row_index * stride + col_index + 2] +
                               (image_buffer[row_index * stride + col_index + 3] << 8) +
                               image_buffer[row_index * stride + col_index + 4] +
                               (image_buffer[row_index * stride + col_index + 5] << 8) ) * 255
                              / (3 * normalize_factor);
                histogram[bin]++;
            }
        }
        break;
    default :
        break;
    }

    input->unmap();

    return true;
}

bool
CLTnrImageKernel::detect_motion (const float* vector_u, const float* vector_v, const uint32_t vector_len, int& delta, float& corr)
{
    delta = 0;
    corr = 0;

    if ( (NULL == vector_u) || (NULL == vector_v) || (0 == vector_len) ) {
        return false;
    }

    int count = vector_len / 6;
    int index = 0;

    float coef = 0;
    const float* p_u = NULL;
    const float* p_v = NULL;
    uint32_t mask_len = 0;

    for ( int i = -count; i <= count;  i++ ) {
        if ( i < 0 ) {
            p_u = &vector_u[abs(i)];
            p_v = &vector_v[0];
            mask_len = vector_len + i;
        }
        else {
            p_u = &vector_u[0];
            p_v = &vector_v[abs(i)];
            mask_len = vector_len - i;
        }

        coef = calculate_correlation(p_u, p_v, mask_len);
        if (corr < coef) {
            corr = coef;
            index = i;
        }
    }

    delta = index;

    return true;
}

/*
    corr(a, b)
    {
        a = a - mean(a);
        b = b - mean(b);
        r = sum(sum(a.*b))/sqrt(sum(sum(a.*a))*sum(sum(b.*b)));
    }

    mean(x) = sum(x(:),'double') / numel(x);
*/
float
CLTnrImageKernel::calculate_correlation (const float* vector_u, const float* vector_v, const uint32_t vector_len)
{
    if ( (NULL == vector_u) || (NULL == vector_v) || (0 == vector_len) ) {
        return 0;
    }

    float uv = 0;
    float uu = 0;
    float vv = 0;
    float u_mean = 0;
    float v_mean = 0;
    float corr = 0;

    for ( uint32_t i = 0; i < vector_len; i++ ) {
        u_mean += vector_u[i];
        v_mean += vector_v[i];
    }

    u_mean = u_mean / vector_len;
    v_mean = v_mean / vector_len;

    for (uint32_t i = 0; i < vector_len; i++) {
        uv += (vector_u[i] - u_mean) * (vector_v[i] - v_mean);
        uu += (vector_u[i] - u_mean) * (vector_u[i] - u_mean);
        vv += (vector_v[i] - v_mean) * (vector_v[i] - v_mean);
    }

    if ((uu != 0) && (vv != 0)) {
        corr = uv / (sqrt(uu) * sqrt(vv));
    }
    else {
        corr = 0.0;
    }

    return corr;
}

void
CLTnrImageKernel::print_image_histogram ()
{
    uint32_t hor_hist_bin = _image_histogram.hor_hist_bin;
    uint32_t ver_hist_bin = _image_histogram.ver_hist_bin;

    XCAM_LOG_DEBUG ("hor hist bin = %d, ver hist bin = %d", hor_hist_bin, ver_hist_bin);

    printf("float hor_hist_current[] = { ");
    for (uint32_t i = 0; i < hor_hist_bin; i++) {
        printf("%f, ", _image_histogram.hor_hist_current[i]);
    }
    printf(" }; \n\n\n");

    printf("float ver_hist_current[] = { ");
    for (uint32_t i = 0; i < ver_hist_bin; i++) {
        printf("%f, ", _image_histogram.ver_hist_current[i]);
    }
    printf(" }; \n\n\n");

    printf("float hor_hist_reference[] = { ");
    for (uint32_t i = 0; i < hor_hist_bin; i++) {
        printf("%f, ", _image_histogram.hor_hist_reference[i]);
    }
    printf(" }; \n\n\n");

    printf("float ver_hist_reference[] = { ");
    for (uint32_t i = 0; i < ver_hist_bin; i++) {
        printf("%f, ", _image_histogram.ver_hist_reference[i]);
    }
    printf(" }; \n\n\n");
}

CLTnrImageHandler::CLTnrImageHandler (const char *name)
    : CLImageHandler (name)
{
}

bool
CLTnrImageHandler::set_tnr_kernel(SmartPtr<CLTnrImageKernel> &kernel)
{
    SmartPtr<CLImageKernel> image_kernel = kernel;
    add_kernel (image_kernel);
    _tnr_kernel = kernel;
    return true;
}

bool
CLTnrImageHandler::set_mode (uint32_t mode)
{
    if (!_tnr_kernel->is_valid ()) {
        XCAM_LOG_ERROR ("set mode error, invalid TNR kernel !");
        return false;
    }

    _tnr_kernel->set_enable (mode & (CL_TNR_TYPE_YUV | CL_TNR_TYPE_RGB));
    return true;
}

bool
CLTnrImageHandler::set_framecount (uint8_t count)
{
    if (!_tnr_kernel->is_valid ()) {
        XCAM_LOG_ERROR ("set framecount error, invalid TNR kernel !");
        return false;
    }

    _tnr_kernel->set_framecount (count);

    return true;
}

bool
CLTnrImageHandler::set_rgb_config (const XCam3aResultTemporalNoiseReduction& config)

{
    if (!_tnr_kernel->is_valid ()) {
        XCAM_LOG_ERROR ("set threshold error, invalid TNR kernel !");
        return false;
    }

    _tnr_kernel->set_rgb_config (config);

    return true;
}

bool
CLTnrImageHandler::set_yuv_config (const XCam3aResultTemporalNoiseReduction& config)

{
    if (!_tnr_kernel->is_valid ()) {
        XCAM_LOG_ERROR ("set threshold error, invalid TNR kernel !");
        return false;
    }

    _tnr_kernel->set_yuv_config (config);

    return true;
}

SmartPtr<CLImageHandler>
create_cl_tnr_image_handler (SmartPtr<CLContext> &context, CLTnrType type)
{
    SmartPtr<CLTnrImageHandler> tnr_handler;
    SmartPtr<CLTnrImageKernel> tnr_kernel;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    XCAM_CL_KERNEL_FUNC_SOURCE_BEGIN(kernel_tnr_yuv)
#include "kernel_tnr_yuv.clx"
    XCAM_CL_KERNEL_FUNC_END;

    XCAM_CL_KERNEL_FUNC_SOURCE_BEGIN(kernel_tnr_rgb)
#include "kernel_tnr_rgb.clx"
    XCAM_CL_KERNEL_FUNC_END;

    if (CL_TNR_TYPE_YUV == type) {
        tnr_kernel = new CLTnrImageKernel (context, "kernel_tnr_yuv", CL_TNR_TYPE_YUV);
        ret = tnr_kernel->load_from_source (kernel_tnr_yuv_body, strlen (kernel_tnr_yuv_body));
    } else if (CL_TNR_TYPE_RGB == type) {
        tnr_kernel = new CLTnrImageKernel (context, "kernel_tnr_rgb", CL_TNR_TYPE_RGB);
        ret = tnr_kernel->load_from_source (kernel_tnr_rgb_body, strlen (kernel_tnr_rgb_body));
    }

    XCAM_FAIL_RETURN (
        WARNING,
        ret == XCAM_RETURN_NO_ERROR,
        NULL,
        "CL image handler(%s) load source failed", tnr_kernel->get_kernel_name());

    tnr_handler = new CLTnrImageHandler ("cl_handler_tnr");
    XCAM_ASSERT (tnr_kernel->is_valid ());
    tnr_handler->set_tnr_kernel (tnr_kernel);

    return tnr_handler;
}

};
