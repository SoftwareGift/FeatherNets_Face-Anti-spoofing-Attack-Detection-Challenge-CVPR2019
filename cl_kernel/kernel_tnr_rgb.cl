/*
 * function: kernel_tnr_rgb
 *     Temporal Noise Reduction
 * outputFrame:      image2d_t as write only
 * thr:              Motion sensitivity, higher value can cause more motion blur
 * frameCount:       input frame count to be processed
 * inputFrame:       image2d_t as read only
 */

" __kernel void kernel_tnr_rgb(__write_only image2d_t outputFrame, float thr, unsigned char frameCount, __read_only image2d_t inputFrame0, __read_only image2d_t inputFrame1, __read_only image2d_t inputFrame2) "
"{                     "
"    int x = get_global_id(0);                        "
"    int y = get_global_id(1);                        "
"                 "
"    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE |CLK_FILTER_NEAREST;  "
"    float4 pixel_in0;                                                        "
"    float4 pixel_in1;                                                        "
"    float4 pixel_in2;                                                        "
"                      "
"    pixel_in0 = read_imagef(inputFrame0, sampler, (int2)(x, y));             "
"    pixel_in1 = read_imagef(inputFrame1, sampler, (int2)(x, y));             "
"    pixel_in2 = read_imagef(inputFrame2, sampler, (int2)(x, y));             "
"                      "
"    float4 pixel_out;                                     "
"    pixel_out.x = (pixel_in0.x + pixel_in1.x + pixel_in2.x) / 3.0f;                "
"    pixel_out.y = (pixel_in0.y + pixel_in1.y + pixel_in2.y) / 3.0f;                "
"    pixel_out.z = (pixel_in0.z + pixel_in1.z + pixel_in2.z) / 3.0f;                "
"    pixel_out.w = (pixel_in0.w + pixel_in1.w + pixel_in2.w) / 3.0f;                "
"               "
"    write_imagef(outputFrame, (int2)(x, y), pixel_out);   "
"}              "
