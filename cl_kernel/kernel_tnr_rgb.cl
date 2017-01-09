/*
 * function: kernel_tnr_rgb
 *     Temporal Noise Reduction
 * outputFrame:      image2d_t as write only
 * thr:              Motion sensitivity, higher value can cause more motion blur
 * frameCount:       input frame count to be processed
 * inputFrame:       image2d_t as read only
 */

__kernel void kernel_tnr_rgb(__write_only image2d_t outputFrame, float tnr_gain, float thr_r, float thr_g, float thr_b, unsigned char frameCount, __read_only image2d_t inputFrame0, __read_only image2d_t inputFrame1, __read_only image2d_t inputFrame2, __read_only image2d_t inputFrame3)
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

        var.x = (fabs(pixel_in0.x - pixel_in1.x) + fabs(pixel_in1.x - pixel_in2.x) + fabs(pixel_in2.x - pixel_in3.x)) / 3.0f;
        var.y = (fabs(pixel_in0.y - pixel_in1.y) + fabs(pixel_in1.y - pixel_in2.y) + fabs(pixel_in2.y - pixel_in3.y)) / 3.0f;
        var.z = (fabs(pixel_in0.z - pixel_in1.z) + fabs(pixel_in1.z - pixel_in2.z) + fabs(pixel_in2.z - pixel_in3.z)) / 3.0f;

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
