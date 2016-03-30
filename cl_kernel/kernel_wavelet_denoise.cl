
/*
 * function: kernel_wavelet_denoise
 *     wavelet filter for denoise usage
 * in:        input image data as read only
 * threshold:   noise threshold
 * low:
 */

__constant float threshConst[5] = { 50.430166, 20.376415, 10.184031, 6.640919, 3.367972 };

__kernel void kernel_wavelet_denoise(__global uint *src, __global uint *approxOut, __global float *details, __global uint *dest,
                                     int inputYOffset, int outputYOffset, uint inputUVOffset, uint outputUVOffset,
                                     int layer, int decomLevels, float hardThresh, float softThresh)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    size_t width = get_global_size(0);
    size_t height = get_global_size(1);

    int imageWidth = width * 16;
    int imageHeight = height;

    float stdev = 0.0;
    float thold = 0.0;
    float16 deviation = (float16)0.0;

    layer = (layer > 1) ? layer : 1;
    layer = (layer < decomLevels) ? layer : decomLevels;

    src += inputYOffset;
    dest += outputYOffset;

#if WAVELET_DENOISE_UV
    int xScaler = pown(2.0, layer);
    int yScaler = pown(2.0, (layer - 1));
#else
    int xScaler = pown(2.0, (layer - 1));
    int yScaler = xScaler;
#endif

    xScaler = ((x == 0) || (x > imageWidth / 16 - xScaler)) ? 0 : xScaler;
    yScaler = ((y < yScaler) || (y > imageHeight - yScaler)) ? 0 : yScaler;

    uint4 approx;
    float16 detail;

#if WAVELET_DENOISE_UV
    int srcOffset = (layer % 2) ? (inputUVOffset * imageWidth / 4) : 0;
    __global uchar *src_p = (__global uchar *)(src + srcOffset);
#else
    __global uchar *src_p = (__global uchar *)(src);
#endif

    int pixel_index = x * 16 + y * imageWidth;
    int group_index = x * 4 + y * (imageWidth / 4);

#if WAVELET_DENOISE_UV
    uint4 luma;
    int luma_index0 = x * 4 + (2 * y) * (imageWidth / 4);
    int luma_index1 = x * 4 + (2 * y + 1) * (imageWidth / 4);
#else
    uint4 chroma;
    int chroma_index = x * 4 + (y / 2) * (imageWidth / 4);
#endif

    ushort16 a;
    ushort16 b;
    ushort16 c;
    ushort16 d;
    ushort16 e;
    ushort16 f;
    ushort16 g;
    ushort16 h;
    ushort16 i;

    float div = 1.0f / 16.0f;

    a = (ushort16)(convert_ushort(src_p[pixel_index - yScaler * imageWidth - xScaler]), convert_ushort(src_p[pixel_index - yScaler * imageWidth - xScaler + 1]),
                   convert_ushort(src_p[pixel_index - yScaler * imageWidth - xScaler + 2]), convert_ushort(src_p[pixel_index - yScaler * imageWidth - xScaler + 3]),
                   convert_ushort(src_p[pixel_index - yScaler * imageWidth - xScaler + 4]), convert_ushort(src_p[pixel_index - yScaler * imageWidth - xScaler + 5]),
                   convert_ushort(src_p[pixel_index - yScaler * imageWidth - xScaler + 6]), convert_ushort(src_p[pixel_index - yScaler * imageWidth - xScaler + 7]),
                   convert_ushort(src_p[pixel_index - yScaler * imageWidth - xScaler + 8]), convert_ushort(src_p[pixel_index - yScaler * imageWidth - xScaler + 9]),
                   convert_ushort(src_p[pixel_index - yScaler * imageWidth - xScaler + 10]), convert_ushort(src_p[pixel_index - yScaler * imageWidth - xScaler + 11]),
                   convert_ushort(src_p[pixel_index - yScaler * imageWidth - xScaler + 12]), convert_ushort(src_p[pixel_index - yScaler * imageWidth - xScaler + 13]),
                   convert_ushort(src_p[pixel_index - yScaler * imageWidth - xScaler + 14]), convert_ushort(src_p[pixel_index - yScaler * imageWidth - xScaler + 15])
                  );

    b = (ushort16)(convert_ushort(src_p[pixel_index - yScaler * imageWidth]), convert_ushort(src_p[pixel_index - yScaler * imageWidth + 1]),
                   convert_ushort(src_p[pixel_index - yScaler * imageWidth + 2]), convert_ushort(src_p[pixel_index - yScaler * imageWidth + 3]),
                   convert_ushort(src_p[pixel_index - yScaler * imageWidth + 4]), convert_ushort(src_p[pixel_index - yScaler * imageWidth + 5]),
                   convert_ushort(src_p[pixel_index - yScaler * imageWidth + 6]), convert_ushort(src_p[pixel_index - yScaler * imageWidth + 7]),
                   convert_ushort(src_p[pixel_index - yScaler * imageWidth + 8]), convert_ushort(src_p[pixel_index - yScaler * imageWidth + 9]),
                   convert_ushort(src_p[pixel_index - yScaler * imageWidth + 10]), convert_ushort(src_p[pixel_index - yScaler * imageWidth + 11]),
                   convert_ushort(src_p[pixel_index - yScaler * imageWidth + 12]), convert_ushort(src_p[pixel_index - yScaler * imageWidth + 13]),
                   convert_ushort(src_p[pixel_index - yScaler * imageWidth + 14]), convert_ushort(src_p[pixel_index - yScaler * imageWidth + 15])
                  );

    c = (ushort16)(convert_ushort(src_p[pixel_index - yScaler * imageWidth + xScaler]), convert_ushort(src_p[pixel_index - yScaler * imageWidth + xScaler + 1]),
                   convert_ushort(src_p[pixel_index - yScaler * imageWidth + xScaler + 2]), convert_ushort(src_p[pixel_index - yScaler * imageWidth + xScaler + 3]),
                   convert_ushort(src_p[pixel_index - yScaler * imageWidth + xScaler + 4]), convert_ushort(src_p[pixel_index - yScaler * imageWidth + xScaler + 5]),
                   convert_ushort(src_p[pixel_index - yScaler * imageWidth + xScaler + 6]), convert_ushort(src_p[pixel_index - yScaler * imageWidth + xScaler + 7]),
                   convert_ushort(src_p[pixel_index - yScaler * imageWidth + xScaler + 8]), convert_ushort(src_p[pixel_index - yScaler * imageWidth + xScaler + 9]),
                   convert_ushort(src_p[pixel_index - yScaler * imageWidth + xScaler + 10]), convert_ushort(src_p[pixel_index - yScaler * imageWidth + xScaler + 11]),
                   convert_ushort(src_p[pixel_index - yScaler * imageWidth + xScaler + 12]), convert_ushort(src_p[pixel_index - yScaler * imageWidth + xScaler + 13]),
                   convert_ushort(src_p[pixel_index - yScaler * imageWidth + xScaler + 14]), convert_ushort(src_p[pixel_index - yScaler * imageWidth + xScaler + 15])
                  );

    d = (ushort16)(convert_ushort(src_p[pixel_index - xScaler]), convert_ushort(src_p[pixel_index - xScaler + 1]),
                   convert_ushort(src_p[pixel_index - xScaler + 2]), convert_ushort(src_p[pixel_index - xScaler + 3]),
                   convert_ushort(src_p[pixel_index - xScaler + 4]), convert_ushort(src_p[pixel_index - xScaler + 5]),
                   convert_ushort(src_p[pixel_index - xScaler + 6]), convert_ushort(src_p[pixel_index - xScaler + 7]),
                   convert_ushort(src_p[pixel_index - xScaler + 8]), convert_ushort(src_p[pixel_index - xScaler + 9]),
                   convert_ushort(src_p[pixel_index - xScaler + 10]), convert_ushort(src_p[pixel_index - xScaler + 11]),
                   convert_ushort(src_p[pixel_index - xScaler + 12]), convert_ushort(src_p[pixel_index - xScaler + 13]),
                   convert_ushort(src_p[pixel_index - xScaler + 14]), convert_ushort(src_p[pixel_index - xScaler + 15])
                  );

    e = (ushort16)(convert_ushort(src_p[pixel_index]), convert_ushort(src_p[pixel_index + 1]),
                   convert_ushort(src_p[pixel_index + 2]), convert_ushort(src_p[pixel_index + 3]),
                   convert_ushort(src_p[pixel_index + 4]), convert_ushort(src_p[pixel_index + 5]),
                   convert_ushort(src_p[pixel_index + 6]), convert_ushort(src_p[pixel_index + 7]),
                   convert_ushort(src_p[pixel_index + 8]), convert_ushort(src_p[pixel_index + 9]),
                   convert_ushort(src_p[pixel_index + 10]), convert_ushort(src_p[pixel_index + 11]),
                   convert_ushort(src_p[pixel_index + 12]), convert_ushort(src_p[pixel_index + 13]),
                   convert_ushort(src_p[pixel_index + 14]), convert_ushort(src_p[pixel_index + 15])
                  );

    f = (ushort16)(convert_ushort(src_p[pixel_index + xScaler]), convert_ushort(src_p[pixel_index + xScaler + 1]),
                   convert_ushort(src_p[pixel_index + xScaler + 2]), convert_ushort(src_p[pixel_index + xScaler + 3]),
                   convert_ushort(src_p[pixel_index + xScaler + 4]), convert_ushort(src_p[pixel_index + xScaler + 5]),
                   convert_ushort(src_p[pixel_index + xScaler + 6]), convert_ushort(src_p[pixel_index + xScaler + 7]),
                   convert_ushort(src_p[pixel_index + xScaler + 8]), convert_ushort(src_p[pixel_index + xScaler + 9]),
                   convert_ushort(src_p[pixel_index + xScaler + 10]), convert_ushort(src_p[pixel_index + xScaler + 11]),
                   convert_ushort(src_p[pixel_index + xScaler + 12]), convert_ushort(src_p[pixel_index + xScaler + 13]),
                   convert_ushort(src_p[pixel_index + xScaler + 14]), convert_ushort(src_p[pixel_index + xScaler + 15])
                  );

    g = (ushort16)(convert_ushort(src_p[pixel_index + yScaler * imageWidth - xScaler]), convert_ushort(src_p[pixel_index + yScaler * imageWidth - xScaler + 1]),
                   convert_ushort(src_p[pixel_index + yScaler * imageWidth - xScaler + 2]), convert_ushort(src_p[pixel_index + yScaler * imageWidth - xScaler + 3]),
                   convert_ushort(src_p[pixel_index + yScaler * imageWidth - xScaler + 4]), convert_ushort(src_p[pixel_index + yScaler * imageWidth - xScaler + 5]),
                   convert_ushort(src_p[pixel_index + yScaler * imageWidth - xScaler + 6]), convert_ushort(src_p[pixel_index + yScaler * imageWidth - xScaler + 7]),
                   convert_ushort(src_p[pixel_index + yScaler * imageWidth - xScaler + 8]), convert_ushort(src_p[pixel_index + yScaler * imageWidth - xScaler + 9]),
                   convert_ushort(src_p[pixel_index + yScaler * imageWidth - xScaler + 10]), convert_ushort(src_p[pixel_index + yScaler * imageWidth - xScaler + 11]),
                   convert_ushort(src_p[pixel_index + yScaler * imageWidth - xScaler + 12]), convert_ushort(src_p[pixel_index + yScaler * imageWidth - xScaler + 13]),
                   convert_ushort(src_p[pixel_index + yScaler * imageWidth - xScaler + 14]), convert_ushort(src_p[pixel_index + yScaler * imageWidth - xScaler + 15])
                  );

    h = (ushort16)(convert_ushort(src_p[pixel_index + yScaler * imageWidth]), convert_ushort(src_p[pixel_index + yScaler * imageWidth + 1]),
                   convert_ushort(src_p[pixel_index + yScaler * imageWidth + 2]), convert_ushort(src_p[pixel_index + yScaler * imageWidth + 3]),
                   convert_ushort(src_p[pixel_index + yScaler * imageWidth + 4]), convert_ushort(src_p[pixel_index + yScaler * imageWidth + 5]),
                   convert_ushort(src_p[pixel_index + yScaler * imageWidth + 6]), convert_ushort(src_p[pixel_index + yScaler * imageWidth + 7]),
                   convert_ushort(src_p[pixel_index + yScaler * imageWidth + 8]), convert_ushort(src_p[pixel_index + yScaler * imageWidth + 9]),
                   convert_ushort(src_p[pixel_index + yScaler * imageWidth + 10]), convert_ushort(src_p[pixel_index + yScaler * imageWidth + 11]),
                   convert_ushort(src_p[pixel_index + yScaler * imageWidth + 12]), convert_ushort(src_p[pixel_index + yScaler * imageWidth + 13]),
                   convert_ushort(src_p[pixel_index + yScaler * imageWidth + 14]), convert_ushort(src_p[pixel_index + yScaler * imageWidth + 15])
                  );

    i = (ushort16)(convert_ushort(src_p[pixel_index + yScaler * imageWidth + xScaler]), convert_ushort(src_p[pixel_index + yScaler * imageWidth + xScaler + 1]),
                   convert_ushort(src_p[pixel_index + yScaler * imageWidth + xScaler + 2]), convert_ushort(src_p[pixel_index + yScaler * imageWidth + xScaler + 3]),
                   convert_ushort(src_p[pixel_index + yScaler * imageWidth + xScaler + 4]), convert_ushort(src_p[pixel_index + yScaler * imageWidth + xScaler + 5]),
                   convert_ushort(src_p[pixel_index + yScaler * imageWidth + xScaler + 6]), convert_ushort(src_p[pixel_index + yScaler * imageWidth + xScaler + 7]),
                   convert_ushort(src_p[pixel_index + yScaler * imageWidth + xScaler + 8]), convert_ushort(src_p[pixel_index + yScaler * imageWidth + xScaler + 9]),
                   convert_ushort(src_p[pixel_index + yScaler * imageWidth + xScaler + 10]), convert_ushort(src_p[pixel_index + yScaler * imageWidth + xScaler + 11]),
                   convert_ushort(src_p[pixel_index + yScaler * imageWidth + xScaler + 12]), convert_ushort(src_p[pixel_index + yScaler * imageWidth + xScaler + 13]),
                   convert_ushort(src_p[pixel_index + yScaler * imageWidth + xScaler + 14]), convert_ushort(src_p[pixel_index + yScaler * imageWidth + xScaler + 15])
                  );

    /*
     { a, b, c } { 1, 2, 1 }
     { d, e, f } { 2, 4, 2 }
     { g, h, i } { 1, 2, 1 }
    */
    ushort16 sum;
    sum = (ushort16)1 * a + (ushort16)2 * b + (ushort16)1 * c +
          (ushort16)2 * d + (ushort16)4 * e + (ushort16)2 * f +
          (ushort16)1 * g + (ushort16)2 * h + (ushort16)1 * i;

    approx = as_uint4(convert_uchar16(((convert_float16(sum) + 0.5 / div) * div)));
    detail = convert_float16(convert_char16(e) - as_char16(approx));

    thold = hardThresh * threshConst[layer - 1];

    detail = (detail < -thold) ? detail + (thold - thold * softThresh) : detail;
    detail = (detail > thold) ? detail - (thold - thold * softThresh) : detail;
    detail = (detail > -thold && detail < thold) ? detail * softThresh : detail;

    __global float16 *details_p = (__global float16 *)(&details[pixel_index]);
    if (layer == 1) {
        (*details_p) = detail;

#if WAVELET_DENOISE_UV
        // copy Y
        luma = vload4(0, src + luma_index0);
        vstore4(luma, 0, dest + luma_index0);
        luma = vload4(0, src + luma_index1);
        vstore4(luma, 0, dest + luma_index1);
#else
        // copy UV
        if (y % 2 == 0) {
            chroma = vload4(0, src + chroma_index + inputUVOffset * (imageWidth / 4));
            vstore4(chroma, 0, dest + chroma_index + outputUVOffset * (imageWidth / 4));
        }
#endif
    } else {
        (*details_p) += detail;
    }

    if (layer < decomLevels) {
#if WAVELET_DENOISE_UV
        int approxOffset = (layer % 2) ? 0 : (inputUVOffset * imageWidth / 4);
        (*(__global uint4*)(approxOut + group_index + approxOffset)) = approx;
#else
        (*(__global uint4*)(approxOut + group_index)) = approx;
#endif
    }
    else
    {
        // Reconstruction
#if WAVELET_DENOISE_UV
        __global uint4* dest_p = (__global uint4*)(&dest[group_index + outputUVOffset * imageWidth / 4]);
        (*dest_p) = as_uint4(convert_uchar16(*details_p + convert_float16(as_uchar16(approx))));
#else
        __global uint4* dest_p = (__global uint4*)(&dest[group_index]);
        (*dest_p) = as_uint4(convert_uchar16(*details_p + convert_float16(as_uchar16(approx))));
#endif
    }
}

