/*
 * function: kernel_tnr_yuv
 *     Temporal Noise Reduction
 * inputFrame:      image2d_t as read only
 * inputFrame0:      image2d_t as read only
 * outputFrame:      image2d_t as write only
 * vertical_offset:  vertical offset from y to uv
 * gain:             Blending ratio of previous and current frame
 * thr_y:            Motion sensitivity for Y, higher value can cause more motion blur
 * thr_uv:            Motion sensitivity for UV, higher value can cause more motion blur
 */

__kernel void kernel_tnr_yuv(
    __read_only image2d_t inputFrame, __read_only image2d_t inputFrame0,
    __write_only image2d_t outputFrame, uint vertical_offset, float gain, float thr_y, float thr_uv)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
    float4 pixel_t0_Y1 = read_imagef(inputFrame0, sampler, (int2)(2 * x, 2 * y));
    float4 pixel_t0_Y2 = read_imagef(inputFrame0, sampler, (int2)(2 * x + 1, 2 * y));
    float4 pixel_t0_Y3 = read_imagef(inputFrame0, sampler, (int2)(2 * x, 2 * y + 1));
    float4 pixel_t0_Y4 = read_imagef(inputFrame0, sampler, (int2)(2 * x + 1, 2 * y + 1));

    float4 pixel_t0_U = read_imagef(inputFrame0, sampler, (int2)(2 * x, y + vertical_offset));
    float4 pixel_t0_V = read_imagef(inputFrame0, sampler, (int2)(2 * x + 1, y + vertical_offset));

    float4 pixel_Y1 = read_imagef(inputFrame, sampler, (int2)(2 * x, 2 * y));
    float4 pixel_Y2 = read_imagef(inputFrame, sampler, (int2)(2 * x + 1, 2 * y));
    float4 pixel_Y3 = read_imagef(inputFrame, sampler, (int2)(2 * x, 2 * y + 1));
    float4 pixel_Y4 = read_imagef(inputFrame, sampler, (int2)(2 * x + 1, 2 * y + 1));

    float4 pixel_U = read_imagef(inputFrame, sampler, (int2)(2 * x, y + vertical_offset));
    float4 pixel_V = read_imagef(inputFrame, sampler, (int2)(2 * x + 1, y + vertical_offset));

    float diff_max = 0.8f;

    float diff_Y = 0.25f * (fabs(pixel_Y1.x - pixel_t0_Y1.x) + fabs(pixel_Y2.x - pixel_t0_Y2.x) +
                            fabs(pixel_Y3.x - pixel_t0_Y3.x) + fabs(pixel_Y4.x - pixel_t0_Y4.x));

    float coeff_Y = (diff_Y < thr_y) ? gain :
                    (diff_Y * (1 - gain) + diff_max * gain - thr_y) / (diff_max - thr_y);
    coeff_Y = (coeff_Y < 1.0f) ? coeff_Y : 1.0f;

    float4 pixel_outY1;
    float4 pixel_outY2;
    float4 pixel_outY3;
    float4 pixel_outY4;
    // X'(K) = (1 - gain) * X'(k-1) + gain * X(k)
    pixel_outY1.x = pixel_t0_Y1.x + (pixel_Y1.x - pixel_t0_Y1.x) * coeff_Y;
    pixel_outY2.x = pixel_t0_Y2.x + (pixel_Y2.x - pixel_t0_Y2.x) * coeff_Y;
    pixel_outY3.x = pixel_t0_Y3.x + (pixel_Y3.x - pixel_t0_Y3.x) * coeff_Y;
    pixel_outY4.x = pixel_t0_Y4.x + (pixel_Y4.x - pixel_t0_Y4.x) * coeff_Y;

    float diff_U = fabs(pixel_U.x - pixel_t0_U.x);
    float diff_V = fabs(pixel_V.x - pixel_t0_V.x);

    float coeff_U = (diff_U < thr_uv) ? gain :
                    (diff_U * (1 - gain) + diff_max * gain - thr_uv) / (diff_max - thr_uv);
    float coeff_V = (diff_V < thr_uv) ? gain :
                    (diff_V * (1 - gain) + diff_max * gain - thr_uv) / (diff_max - thr_uv);
    coeff_U = (coeff_U < 1.0f) ? coeff_U : 1.0f;
    coeff_V = (coeff_V < 1.0f) ? coeff_V : 1.0f;

    float4 pixel_outU;
    float4 pixel_outV;
    pixel_outU.x = pixel_t0_U.x + (pixel_U.x - pixel_t0_U.x) * coeff_U;
    pixel_outV.x = pixel_t0_V.x + (pixel_V.x - pixel_t0_V.x) * coeff_V;

    write_imagef(outputFrame, (int2)(2 * x, 2 * y), pixel_outY1);
    write_imagef(outputFrame, (int2)(2 * x + 1, 2 * y), pixel_outY2);
    write_imagef(outputFrame, (int2)(2 * x, 2 * y + 1), pixel_outY3);
    write_imagef(outputFrame, (int2)(2 * x + 1, 2 * y + 1), pixel_outY4);
    write_imagef(outputFrame, (int2)(2 * x, y + vertical_offset), pixel_outU);
    write_imagef(outputFrame, (int2)(2 * x + 1, y + vertical_offset), pixel_outV);
}

/*
 * function: kernel_tnr_rgb
 *     Temporal Noise Reduction
 * outputFrame:      image2d_t as write only
 * thr:              Motion sensitivity, higher value can cause more motion blur
 * frameCount:       input frame count to be processed
 * inputFrame:       image2d_t as read only
 */

__kernel void kernel_tnr_rgb(
    __write_only image2d_t outputFrame,
    float tnr_gain, float thr_r, float thr_g, float thr_b, unsigned char frameCount,
    __read_only image2d_t inputFrame0, __read_only image2d_t inputFrame1,
    __read_only image2d_t inputFrame2, __read_only image2d_t inputFrame3)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

    float4 pixel_in0;
    float4 pixel_in1;
    float4 pixel_in2;
    float4 pixel_in3;

    float4 pixel_out;
    float4 var;
    float gain = 0;
    int cond;

    pixel_in0 =  read_imagef(inputFrame0, sampler, (int2)(x, y));
    pixel_in1 =  read_imagef(inputFrame1, sampler, (int2)(x, y));

    if(frameCount == 4) {
        pixel_in2 =  read_imagef(inputFrame2, sampler, (int2)(x, y));
        pixel_in3 =  read_imagef(inputFrame3, sampler, (int2)(x, y));

        var.x = (fabs(pixel_in0.x - pixel_in1.x) + fabs(pixel_in1.x - pixel_in2.x) +
                 fabs(pixel_in2.x - pixel_in3.x)) / 3.0f;
        var.y = (fabs(pixel_in0.y - pixel_in1.y) + fabs(pixel_in1.y - pixel_in2.y) +
                 fabs(pixel_in2.y - pixel_in3.y)) / 3.0f;
        var.z = (fabs(pixel_in0.z - pixel_in1.z) + fabs(pixel_in1.z - pixel_in2.z) +
                 fabs(pixel_in2.z - pixel_in3.z)) / 3.0f;

        cond = (var.x + var.y + var.z) < (thr_r + thr_g + thr_b);
        gain = cond ? 1.0f : 0.0f;

        pixel_out.x = (gain * pixel_in0.x + gain * pixel_in1.x + gain * pixel_in2.x + pixel_in3.x) / (1.0f + 3 * gain);
        pixel_out.y = (gain * pixel_in0.y + gain * pixel_in1.y + gain * pixel_in2.y + pixel_in3.y) / (1.0f + 3 * gain);
        pixel_out.z = (gain * pixel_in0.z + gain * pixel_in1.z + gain * pixel_in2.z + pixel_in3.z) / (1.0f + 3 * gain);
    }
    else if(frameCount == 3) {
        pixel_in2 =  read_imagef(inputFrame2, sampler, (int2)(x, y));
        var.x = (fabs(pixel_in0.x - pixel_in1.x) + fabs(pixel_in1.x - pixel_in2.x)) / 2.0f;
        var.y = (fabs(pixel_in0.y - pixel_in1.y) + fabs(pixel_in1.y - pixel_in2.y)) / 2.0f;
        var.z = (fabs(pixel_in0.z - pixel_in1.z) + fabs(pixel_in1.z - pixel_in2.z)) / 2.0f;

        cond = (var.x + var.y + var.z) < (thr_r + thr_g + thr_b);
        gain = cond ? 1.0f : 0.0f;

        pixel_out.x = (gain * pixel_in0.x + gain * pixel_in1.x + pixel_in2.x) / (1.0f + 2 * gain);
        pixel_out.y = (gain * pixel_in0.y + gain * pixel_in1.y + pixel_in2.y) / (1.0f + 2 * gain);
        pixel_out.z = (gain * pixel_in0.z + gain * pixel_in1.z + pixel_in2.z) / (1.0f + 2 * gain);
    }
    else if(frameCount == 2)
    {
        var.x = fabs(pixel_in0.x - pixel_in1.x);
        var.y = fabs(pixel_in0.y - pixel_in1.y);
        var.z = fabs(pixel_in0.z - pixel_in1.z);

        cond = (var.x + var.y + var.z) < (thr_r + thr_g + thr_b);
        gain = cond ? 1.0f : 0.0f;

        pixel_out.x = (gain * pixel_in0.x + pixel_in1.x) / (1.0f + gain);
        pixel_out.y = (gain * pixel_in0.y + pixel_in1.y) / (1.0f + gain);
        pixel_out.z = (gain * pixel_in0.z + pixel_in1.z) / (1.0f + gain);
    }

    write_imagef(outputFrame, (int2)(x, y), pixel_out);
}

