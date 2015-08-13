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

    pixel_in0 = read_imagef(inputFrame0, sampler, (int2)(x, y));
    pixel_in1 = read_imagef(inputFrame1, sampler, (int2)(x, y));
    pixel_in2 = read_imagef(inputFrame2, sampler, (int2)(x, y));
    pixel_in3 = read_imagef(inputFrame3, sampler, (int2)(x, y));

    float4 pixel_out;
    float4 var;
    float gain = 0;
    var.x = (fabs(pixel_in0.x - pixel_in1.x) + fabs(pixel_in1.x - pixel_in2.x) + fabs(pixel_in2.x - pixel_in3.x)) / 3.0;
    var.y = (fabs(pixel_in0.y - pixel_in1.y) + fabs(pixel_in1.y - pixel_in2.y) + fabs(pixel_in2.y - pixel_in3.y)) / 3.0;
    var.z = (fabs(pixel_in0.z - pixel_in1.z) + fabs(pixel_in1.z - pixel_in2.z) + fabs(pixel_in2.z - pixel_in3.z)) / 3.0;
    if ((var.x + var.y + var.z) < (thr_r + thr_g + thr_b)) {
        gain = 1.0;
    }

    pixel_out.x = (gain * pixel_in0.x + gain * pixel_in1.x + gain * pixel_in2.x + pixel_in3.x) / (1.0f + 3 * gain);
    pixel_out.y = (gain * pixel_in0.y + gain * pixel_in1.y + gain * pixel_in2.y + pixel_in3.y) / (1.0f + 3 * gain);
    pixel_out.z = (gain * pixel_in0.z + gain * pixel_in1.z + gain * pixel_in2.z + pixel_in3.z) / (1.0f + 3 * gain);

    pixel_out.w = (pixel_in0.w + pixel_in1.w + pixel_in2.w + pixel_in3.w) / 4.0f;

    write_imagef(outputFrame, (int2)(x, y), pixel_out);
}
