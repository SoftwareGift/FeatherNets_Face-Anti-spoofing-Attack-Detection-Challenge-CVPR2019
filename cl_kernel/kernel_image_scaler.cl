/**
* \brief Image scaling kernel function.
* \param[in] input Input image object.
* \param[out] output scaled output image object.
* \param[in] output_widht: output width
* \param[in] output_height: output height
* \param[in] vertical_offset:  vertical offset from y to uv
*/
__kernel void kernel_image_scaler (__read_only image2d_t input,
                                   __write_only image2d_t output,
                                   const uint output_widht,
                                   const uint output_height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

    float2 normCoor = convert_float2((int2)(x, y)) / (float2)(output_widht, output_height);
    float4 scaled_pixel = read_imagef(input, sampler, normCoor);
    write_imagef(output, (int2)(x, y), scaled_pixel);
}

