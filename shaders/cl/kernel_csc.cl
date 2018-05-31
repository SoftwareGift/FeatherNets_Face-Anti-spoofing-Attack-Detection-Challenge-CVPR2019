/*
 * function: kernel_csc_rgbatonv12
 * input:    image2d_t as read only
 * output:   image2d_t as write only
 * vertical_offset, vertical offset from y to uv
 */

__kernel void kernel_csc_rgbatonv12 (__read_only image2d_t input, __write_only image2d_t output_y, __write_only image2d_t output_uv, __global float *matrix)
{
    int x = get_global_id (0);
    int y = get_global_id (1);
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
    float4 pixel_in1 = read_imagef(input, sampler, (int2)(2 * x, 2 * y));
    float4 pixel_in2 = read_imagef(input, sampler, (int2)(2 * x + 1, 2 * y));
    float4 pixel_in3 = read_imagef(input, sampler, (int2)(2 * x, 2 * y + 1));
    float4 pixel_in4 = read_imagef(input, sampler, (int2)(2 * x + 1, 2 * y + 1));
    float4 pixel_out_y1, pixel_out_y2, pixel_out_y3, pixel_out_y4, pixel_out_u, pixel_out_v;
    pixel_out_y1.x = matrix[0] * pixel_in1.x + matrix[1] * pixel_in1.y + matrix[2] * pixel_in1.z;
    pixel_out_y1.y = 0.0f;
    pixel_out_y1.z = 0.0f;
    pixel_out_y1.w = 1.0f;
    pixel_out_y2.x = matrix[0] * pixel_in2.x + matrix[1] * pixel_in2.y +  matrix[2] * pixel_in2.z;
    pixel_out_y2.y = 0.0f;
    pixel_out_y2.z = 0.0f;
    pixel_out_y2.w = 1.0f;
    pixel_out_y3.x = matrix[0] * pixel_in3.x + matrix[1] * pixel_in3.y + matrix[2] * pixel_in3.z;
    pixel_out_y3.y = 0.0f;
    pixel_out_y3.z = 0.0f;
    pixel_out_y3.w = 1.0f;
    pixel_out_y4.x = matrix[0] * pixel_in4.x + matrix[1] * pixel_in4.y + matrix[2] * pixel_in4.z;
    pixel_out_y4.y = 0.0f;
    pixel_out_y4.z = 0.0f;
    pixel_out_y4.w = 1.0f;
    pixel_out_u.x = matrix[3] * pixel_in1.x + matrix[4] * pixel_in1.y + matrix[5] * pixel_in1.z + 0.5f;
    pixel_out_u.y = 0.0f;
    pixel_out_u.z = 0.0f;
    pixel_out_u.w = 1.0f;
    pixel_out_v.x = matrix[6] * pixel_in1.x + matrix[7] * pixel_in1.y + matrix[8] * pixel_in1.z + 0.5f;
    pixel_out_v.y = 0.0f;
    pixel_out_v.z = 0.0f;
    pixel_out_v.w = 1.0f;
    write_imagef(output_y, (int2)(2 * x, 2 * y), pixel_out_y1);
    write_imagef(output_y, (int2)(2 * x + 1, 2 * y), pixel_out_y2);
    write_imagef(output_y, (int2)(2 * x, 2 * y + 1), pixel_out_y3);
    write_imagef(output_y, (int2)(2 * x + 1, 2 * y + 1), pixel_out_y4);
    write_imagef(output_uv, (int2)(2 * x, y), pixel_out_u);
    write_imagef(output_uv, (int2)(2 * x + 1, y), pixel_out_v);
}


/*
 * function: kernel_csc_rgbatolab
 * input:    image2d_t as read only
 * output:   image2d_t as write only
 */

static float lab_fun(float a)
{
    if (a > 0.008856f)
        return pow(a, 1.0f / 3);
    else
        return (float)(7.787f * a + 16.0f / 116);
}
__kernel void kernel_csc_rgbatolab (__read_only image2d_t input, __write_only image2d_t output)
{
    int x = get_global_id (0);
    int y = get_global_id (1);
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
    float4 pixel_in = read_imagef(input, sampler, (int2)(x, y));
    float X, Y, Z, L, a, b;
    X = 0.433910f * pixel_in.x + 0.376220f * pixel_in.y + 0.189860f * pixel_in.z;
    Y = 0.212649f * pixel_in.x + 0.715169f * pixel_in.y + 0.072182f * pixel_in.z;
    Z = 0.017756f * pixel_in.x + 0.109478f * pixel_in.y + 0.872915f * pixel_in.z;
    if(Y > 0.008856f)
        L = 116 * (pow(Y, 1.0f / 3));
    else
        L = 903.3f * Y;
    a = 500 * (lab_fun(X) - lab_fun(Y));
    b = 200 * (lab_fun(Y) - lab_fun(Z));
    write_imagef(output, (int2)(3 * x, y), L);
    write_imagef(output, (int2)(3 * x + 1, y), a);
    write_imagef(output, (int2)(3 * x + 2, y), b);
}

/*
 * function: kernel_csc_rgba64torgba
 * input:    image2d_t as read only
 * output:   image2d_t as write only
 */
__kernel void kernel_csc_rgba64torgba (__read_only image2d_t input, __write_only image2d_t output)
{
    int x = get_global_id (0);
    int y = get_global_id (1);
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
    float4 pixel_in = read_imagef(input, sampler, (int2)(x, y));
    write_imagef(output, (int2)(x, y), pixel_in);
}

/*
 * function: kernel_csc_yuyvtorgba
 * input:    image2d_t as read only
 * output:   image2d_t as write only
 */

__kernel void kernel_csc_yuyvtorgba (__read_only image2d_t input, __write_only image2d_t output)
{
    int x = get_global_id (0);
    int y = get_global_id (1);
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
    float4 pixel_in1 = read_imagef(input, sampler, (int2)(x, y));
    float4 pixel_out1, pixel_out2;
    pixel_out1.x = pixel_in1.x + 1.13983f * (pixel_in1.w - 0.5f);
    pixel_out1.y = pixel_in1.x - 0.39465f * (pixel_in1.y - 0.5f) - 0.5806f * (pixel_in1.w - 0.5f);
    pixel_out1.z = pixel_in1.x + 2.03211f * (pixel_in1.y - 0.5f);
    pixel_out1.w = 0.0f;
    pixel_out2.x = pixel_in1.z + 1.13983f * (pixel_in1.w - 0.5f);
    pixel_out2.y = pixel_in1.z - 0.39465f * (pixel_in1.y - 0.5f) - 0.5806f * (pixel_in1.w - 0.5f);
    pixel_out2.z = pixel_in1.z + 2.03211f * (pixel_in1.y - 0.5f);
    pixel_out2.w = 0.0f;
    write_imagef(output, (int2)(2 * x, y), pixel_out1);
    write_imagef(output, (int2)(2 * x + 1, y), pixel_out2);
}

/*
 * function: kernel_csc_nv12torgba
 * input:    image2d_t as read only
 * output:   image2d_t as write only
 * vertical_offset, vertical offset from y to uv
 */

__kernel void kernel_csc_nv12torgba (
    __read_only image2d_t input_y, __write_only image2d_t output, __read_only image2d_t input_uv)
{
    int x = get_global_id (0);
    int y = get_global_id (1);
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
    float4 pixel_y1 = read_imagef(input_y, sampler, (int2)(2 * x, 2 * y));
    float4 pixel_y2 = read_imagef(input_y, sampler, (int2)(2 * x + 1, 2 * y));
    float4 pixel_y3 = read_imagef(input_y, sampler, (int2)(2 * x, 2 * y + 1));
    float4 pixel_y4 = read_imagef(input_y, sampler, (int2)(2 * x + 1, 2 * y + 1));
    float4 pixel_u = read_imagef(input_uv, sampler, (int2)(2 * x, y));
    float4 pixel_v = read_imagef(input_uv, sampler, (int2)(2 * x + 1, y));
    float4 pixel_out1, pixel_out2, pixel_out3, pixel_out4;
    pixel_out1.x = pixel_y1.x + 1.13983f * (pixel_v.x - 0.5f);
    pixel_out1.y = pixel_y1.x - 0.39465f * (pixel_u.x - 0.5f) - 0.5806f * (pixel_v.x - 0.5f);
    pixel_out1.z = pixel_y1.x + 2.03211f * (pixel_u.x - 0.5f);
    pixel_out1.w = 0.0f;
    pixel_out2.x = pixel_y2.x + 1.13983f * (pixel_v.x - 0.5f);
    pixel_out2.y = pixel_y2.x - 0.39465f * (pixel_u.x - 0.5f) - 0.5806f * (pixel_v.x - 0.5f);
    pixel_out2.z = pixel_y2.x + 2.03211f * (pixel_u.x - 0.5f);
    pixel_out2.w = 0.0f;
    pixel_out3.x = pixel_y3.x + 1.13983f * (pixel_v.x - 0.5f);
    pixel_out3.y = pixel_y3.x - 0.39465f * (pixel_u.x - 0.5f) - 0.5806f * (pixel_v.x - 0.5f);
    pixel_out3.z = pixel_y3.x + 2.03211f * (pixel_u.x - 0.5f);
    pixel_out3.w = 0.0f;
    pixel_out4.x = pixel_y4.x + 1.13983f * (pixel_v.x - 0.5f);
    pixel_out4.y = pixel_y4.x - 0.39465f * (pixel_u.x - 0.5f) - 0.5806f * (pixel_v.x - 0.5f);
    pixel_out4.z = pixel_y4.x + 2.03211f * (pixel_u.x - 0.5f);
    pixel_out4.w = 0.0f;
    write_imagef(output, (int2)(2 * x, 2 * y), pixel_out1);
    write_imagef(output, (int2)(2 * x + 1, 2 * y), pixel_out2);
    write_imagef(output, (int2)(2 * x, 2 * y + 1), pixel_out3);
    write_imagef(output, (int2)(2 * x + 1, 2 * y + 1), pixel_out4);
}

