/*
 * function: kernel_ee
 * input:    image2d_t as read only
 * output:   image2d_t as write only
 * ee_config: Edge enhancement configuration
 */

typedef struct
{
    float           ee_gain;
    float           ee_threshold;
    float           nr_gain;
} CLEeConfig;

__constant float lv[25] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                           0.0f, 0.0f, -1.0f, 0.0f, 0.0f,
                           -1.0f, -14.0f, 32.0f, -14.0f, -1.0f,
                           0.0f, 0.0f, -1.0f, 0.0f, 0.0f,
                           0.0f, 0.0f, 0.0f, 0.0f, 0.0f
                          };

__constant float lh[25] = {0.0f, 0.0f, -1.0f, 0.0f, 0.0f,
                           0.0f, 0.0f, -14.0f, 0.0f, 0.0f,
                           0.0f, -1.0f, 32.0f, -1.0f, 0.0f,
                           0.0f, 0.0f, -14.0f, 0.0f, 0.0f,
                           0.0f, 0.0f, -1.0f, 0.0f, 0.0f
                          };

__constant float la[25] = {0.0f, 0.0f, -2.0f, 0.0f, 0.0f,
                           0.0f, -2.0f, -2.0f, -2.0f, 0.0f,
                           -2.0f, -2.0f, 24.0f, -2.0f, -2.0f,
                           0.0f, -2.0f, -2.0f, -2.0f, 0.0f,
                           0.0f, 0.0f, -2.0f, 0.0f, 0.0f
                          };

__constant float na[25] = { -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
                            -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
                            -1.0f, -1.0f, 16.0f, -1.0f, -1.0f,
                            -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
                            -1.0f, -1.0f, -1.0f, -1.0f, -1.0f
                          };
__kernel void kernel_ee (__read_only image2d_t input, __write_only image2d_t output, uint vertical_offset_in, uint vertical_offset_out, CLEeConfig ee_config)
{
    int x = get_global_id (0);
    int y = get_global_id (1);
    int X = get_global_size(0);
    int Y = get_global_size(1);

    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    float4 y_in, y_out, uv_in;
    float4 a[5], b[5], c[5], d[5], e[5];

    // cpy UV
    if(y % 2 == 0) {
        uv_in = read_imagef(input, sampler, (int2)(x, y / 2 + vertical_offset_in));
        write_imagef(output, (int2)(x, y / 2 + vertical_offset_out), uv_in);
    }

    a[0] = read_imagef(input, sampler, (int2)(x - 2, y - 2));
    a[1] = read_imagef(input, sampler, (int2)(x - 1, y - 2));
    a[2] = read_imagef(input, sampler, (int2)(x, y - 2));
    a[3] = read_imagef(input, sampler, (int2)(x + 1, y - 2));
    a[4] = read_imagef(input, sampler, (int2)(x + 2, y - 2));

    b[0] = read_imagef(input, sampler, (int2)(x - 2, y - 1));
    b[1] = read_imagef(input, sampler, (int2)(x - 1, y - 1));
    b[2] = read_imagef(input, sampler, (int2)(x, y - 1));
    b[3] = read_imagef(input, sampler, (int2)(x + 1, y - 1));
    b[4] = read_imagef(input, sampler, (int2)(x + 2, y - 1));

    c[0] = read_imagef(input, sampler, (int2)(x - 2, y));
    c[1] = read_imagef(input, sampler, (int2)(x - 1, y));
    c[2] = read_imagef(input, sampler, (int2)(x, y));
    c[3] = read_imagef(input, sampler, (int2)(x + 1, y));
    c[4] = read_imagef(input, sampler, (int2)(x + 2, y));

    d[0] = read_imagef(input, sampler, (int2)(x - 2, y + 1));
    d[1] = read_imagef(input, sampler, (int2)(x - 1, y + 1));
    d[2] = read_imagef(input, sampler, (int2)(x, y + 1));
    d[3] = read_imagef(input, sampler, (int2)(x + 1, y + 1));
    d[4] = read_imagef(input, sampler, (int2)(x + 2, y + 1));

    e[0] = read_imagef(input, sampler, (int2)(x - 2, y + 2));
    e[1] = read_imagef(input, sampler, (int2)(x - 1, y + 2));
    e[2] = read_imagef(input, sampler, (int2)(x, y + 2));
    e[3] = read_imagef(input, sampler, (int2)(x + 1, y + 2));
    e[4] = read_imagef(input, sampler, (int2)(x + 2, y + 2));

    float eV = (a[0].x * lv[0] + a[1].x * lv[1] + a[2].x * lv[2] + a[3].x * lv[3] + a[4].x * lv[4]
                + b[0].x * lv[5] + b[1].x * lv[6] + b[2].x * lv[7] + b[3].x * lv[8] + b[4].x * lv[9]
                + c[0].x * lv[10] + c[1].x * lv[11] + c[2].x * lv[12] + c[3].x * lv[13] + c[4].x * lv[14]
                + d[0].x * lv[15] + d[1].x * lv[16] + d[2].x * lv[17] + d[3].x * lv[18] + d[4].x * lv[19]
                + e[0].x * lv[20] + e[1].x * lv[21] + e[2].x * lv[22] + e[3].x * lv[23] + e[4].x * lv[24]) * 255.0f;

    float eH = (a[0].x * lh[0] + a[1].x * lh[1] + a[2].x * lh[2] + a[3].x * lh[3] + a[4].x * lh[4]
                + b[0].x * lh[5] + b[1].x * lh[6] + b[2].x * lh[7] + b[3].x * lh[8] + b[4].x * lh[9]
                + c[0].x * lh[10] + c[1].x * lh[11] + c[2].x * lh[12] + c[3].x * lh[13] + c[4].x * lh[14]
                + d[0].x * lh[15] + d[1].x * lh[16] + d[2].x * lh[17] + d[3].x * lh[18] + d[4].x * lh[19]
                + e[0].x * lh[20] + e[1].x * lh[21] + e[2].x * lh[22] + e[3].x * lh[23] + e[4].x * lh[24]) * 255.0f;


    float eA = (a[0].x * la[0] + a[1].x * la[1] + a[2].x * la[2] + a[3].x * la[3] + a[4].x * la[4]
                + b[0].x * la[5] + b[1].x * la[6] + b[2].x * la[7] + b[3].x * la[8] + b[4].x * la[9]
                + c[0].x * la[10] + c[1].x * la[11] + c[2].x * la[12] + c[3].x * la[13] + c[4].x * la[14]
                + d[0].x * la[15] + d[1].x * la[16] + d[2].x * la[17] + d[3].x * la[18] + d[4].x * la[19]
                + e[0].x * la[20] + e[1].x * la[21] + e[2].x * la[22] + e[3].x * la[23] + e[4].x * la[24]) * 255.0f;

    float nA = (a[0].x * na[0] + a[1].x * na[1] + a[2].x * na[2] + a[3].x * na[3] + a[4].x * na[4]
                + b[0].x * na[5] + b[1].x * na[6] + b[2].x * na[7] + b[3].x * na[8] + b[4].x * na[9]
                + c[0].x * na[10] + c[1].x * na[11] + c[2].x * na[12] + c[3].x * na[13] + c[4].x * na[14]
                + d[0].x * na[15] + d[1].x * na[16] + d[2].x * na[17] + d[3].x * na[18] + d[4].x * na[19]
                + e[0].x * na[20] + e[1].x * na[21] + e[2].x * na[22] + e[3].x * na[23] + e[4].x * na[24]) * 255.0f;


    float nV = eH;
    float nH = eV;

    float dV = (fabs(2.0f * b[1].x - a[1].x - c[1].x) + fabs(2.0f * b[2].x - a[2].x - c[2].x) + fabs(2.0f * b[3].x - a[3].x - c[3].x) + fabs(2.0f * c[1].x - b[1].x - d[1].x) + fabs(2.0f * c[2].x - b[2].x - d[2].x) + fabs(2.0f * c[3].x - b[3].x - d[3].x) + fabs(2.0f * d[1].x - c[1].x - e[1].x) + fabs(2.0f * d[2].x - c[2].x - e[2].x) + fabs(2.0f * d[3].x - c[3].x - e[3].x)) * 255.0f;
    float dH = (fabs(2.0f * b[1].x - b[0].x - b[2].x) + fabs(2.0f * b[2].x - b[1].x - b[3].x) + fabs(2.0f * b[3].x - b[2].x - b[4].x) + fabs(2.0f * c[1].x - c[0].x - c[2].x) + fabs(2.0f * c[2].x - c[1].x - c[3].x) + fabs(2.0f * c[3].x - c[2].x - c[4].x) + fabs(2.0f * d[1].x - d[0].x - d[2].x) + fabs(2.0f * d[2].x - d[1].x - d[3].x) + fabs(2.0f * d[3].x - d[2].x - d[4].x)) * 255.0f;
    float dA = (fabs(2.0f * c[2].x - b[2].x - d[2].x) + fabs(2.0f * c[2].x - c[1].x - c[3].x) + fabs(2.0f * c[2].x - b[1].x - d[3].x) + fabs(2.0f * c[2].x - b[3].x - d[1].x) + fabs(2.0f * c[2].x - a[0].x - e[4].x) + fabs(2.0f * c[2].x - a[4].x - e[0].x) + fabs(2.0f * c[2].x - c[0].x - c[4].x) + fabs(2.0f * c[2].x - a[2].x - e[2].x) + fabs(2.0f * c[2].x - (b[0].x + d[0].x + b[4].x + d[4].x) / 2.0f)) * 255.0f;

    float edge = dH < (dV < dA ? dV : dA) ? eH  : (dV < dA ? eV : eA);
    float noise = dH < (dV < dA ? dV : dA) ? nH  : (dV < dA ? nV : nA);
    float dir = dH < (dV < dA ? dA : dV) ? (dV < dA ? dA : dV) : dH;
    noise = noise * ee_config.nr_gain / 16.0f;
    edge = edge * ee_config.ee_gain / 16.0f;

    y_out.x = dir > ee_config.ee_threshold ? (c[2].x * 255.0f + edge - noise) / 255.0f : c[2].x;
    y_out.y = 0.0f;
    y_out.z = 0.0f;
    y_out.w = 1.0f;
    write_imagef(output, (int2)(x, y), y_out);
}
