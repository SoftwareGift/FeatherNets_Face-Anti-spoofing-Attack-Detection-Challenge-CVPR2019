/*
 * function: kernel_dpc
 *     defect pixel correction on bayer data input
 * input:    image2d_t as read only
 * output:   image2d_t as write only
 * gr_threshold:   GR threshold of defect pixel correction
 * r_threshold:    R threshold of defect pixel correction
 * b_threshold:    B threshold of defect pixel correction
 * gb_threshold:   GB threshold of defect pixel correction
 * param:
 */

__kernel void kernel_dpc (__read_only image2d_t input,
                          __write_only image2d_t output,
                          float gr_threshold, float r_threshold,
                          float b_threshold, float gb_threshold)
{
    int x = get_global_id (0);
    int y = get_global_id (1);
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    float4 p[9];
    p[0] = read_imagef(input, sampler, (int2)(x - 2, y - 2));
    p[1] = read_imagef(input, sampler, (int2)(x, y - 2));
    p[2] = read_imagef(input, sampler, (int2)(x + 2, y - 2));
    p[3] = read_imagef(input, sampler, (int2)(x - 2, y));
    p[4] = read_imagef(input, sampler, (int2)(x, y));
    p[5] = read_imagef(input, sampler, (int2)(x + 2, y));
    p[6] = read_imagef(input, sampler, (int2)(x - 2, y + 2));
    p[7] = read_imagef(input, sampler, (int2)(x, y + 2));
    p[8] = read_imagef(input, sampler, (int2)(x + 2, y + 2));

    float aveVer = (p[1].x + p[7].x) / 2;
    float aveHor = (p[3].x + p[5].x) / 2;
    float avePosDia = (p[0].x + p[8].x) / 2;
    float aveNegDia = (p[2].x + p[6].x) / 2;

    float aveMin, aveMax;
    if (aveVer > aveHor)  {
        aveMin = aveHor;
        aveMax = aveVer;
    }
    else {
        aveMin = aveVer;
        aveMax = aveHor;
    }

    if (avePosDia < aveMin)
        aveMin = avePosDia;
    else if (avePosDia > aveMax)
        aveMax = avePosDia;

    if (aveNegDia < aveMin)
        aveMin = aveNegDia;
    else if (aveNegDia > aveMax)
        aveMax = aveNegDia;

    float edgeVer = p[4].x - aveVer;
    float edgeHor = p[4].x - aveHor;
    float edgeNeighbourVer = (p[3].x + p[5].x - (p[0].x + p[2].x + p[6].x + p[8].x) / 2) / 2;
    float edgeNeighbourHor = (p[1].x + p[7].x - (p[0].x + p[2].x + p[6].x + p[8].x) / 2) / 2;

    float threshold;
    if (x % 2 == 0)
        threshold = (y % 2 == 0) ? gr_threshold : b_threshold;
    else
        threshold = (y % 2 == 0) ? r_threshold : gb_threshold;

    float4 pixelOut;
    pixelOut.x = p[4].x;
    pixelOut.y = p[4].y;
    pixelOut.z = p[4].z;
    pixelOut.w = p[4].w;
    if ((edgeVer > edgeNeighbourVer) && (edgeHor > edgeNeighbourHor))  {
        if ((p[4].x - aveMax) > threshold)  {
            pixelOut.x = aveMax;
        }
    }
    if ((edgeVer < edgeNeighbourVer) && (edgeHor < edgeNeighbourHor))  {
        if ((aveMin - p[4].x) > threshold)  {
            pixelOut.x = aveMin;
        }
    }

    write_imagef (output, (int2)(x, y), pixelOut);
}
