/*
 * function: kernel_3a_stats
 * input:    image2d_t as read only
 * output:   XCamGridStat, stats results
 */

"typedef struct"
"{"
"    unsigned int avg_y;"

"    unsigned int avg_r;"
"    unsigned int avg_gr;"
"    unsigned int avg_gb;"
"    unsigned int avg_b;"
"    unsigned int valid_wb_count;"

"    unsigned int f_value1;"
"    unsigned int f_value2;"
"} XCamGridStat;\n"

"__kernel void kernel_3a_stats (__read_only image2d_t input, __global XCamGridStat *output)   "
"{                                                                                            "
"    int x = get_global_id (0);                                                               "
"    int y = get_global_id (1);                                                               "
"    int w = get_global_size (0);                                                             "
"    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST; "
"                                                                                             "
"    int x0 = 16 * x;                                                                         "
"    int y0 = 16 * y;                                                                         "
"    float sum_gr = 0.0f, sum_r = 0.0f, sum_b = 0.0f, sum_gb=0.0f;                            "
"    float avg_gr = 0.0f, avg_r = 0.0f, avg_b = 0.0f, avg_gb = 0.0f;                          "
"    int i = 0, j = 0;                                                                        "
"    float count = (16.0 / 2) * (16.0 / 2);                                                   "
"    float4 p[4];                                                                             "

"\n#pragma unroll\n"
"    for (j = 0; j < 16; j += 2) {                                                            "

// grid (0, 0)
"    i = 0;                                                                                   "
"    p[0] = read_imagef (input, sampler, (int2)(x0 + i, y0 + j));                             "
"    p[1] = read_imagef (input, sampler, (int2)(x0 + i, y0 + j + 1));                         "
"    ++i;                                                                                     "
"    p[2] = read_imagef (input, sampler, (int2)(x0 + i, y0 + j));                             "
"    p[3] = read_imagef (input, sampler, (int2)(x0 + i, y0 + j + 1));                         "
"    sum_gr += p[0].x;                                                                        "
"    sum_b += p[1].x;                                                                         "
"    sum_r += p[2].x;                                                                         "
"    sum_gb += p[3].x;                                                                        "

// grid (1, 0)
"    ++i;                                                                                     "
"    p[0] = read_imagef (input, sampler, (int2)(x0 + i, y0 + j));                             "
"    p[1] = read_imagef (input, sampler, (int2)(x0 + i, y0 + j + 1));                         "
"    ++i;                                                                                     "
"    p[2] = read_imagef (input, sampler, (int2)(x0 + i, y0 + j));                             "
"    p[3] = read_imagef (input, sampler, (int2)(x0 + i, y0 + j + 1));                         "
"    sum_gr += p[0].x;                                                                        "
"    sum_b += p[1].x;                                                                         "
"    sum_r += p[2].x;                                                                         "
"    sum_gb += p[3].x;                                                                        "

// grid (2, 0)
"    ++i;                                                                                      "
"    p[0] = read_imagef (input, sampler, (int2)(x0 + i, y0 + j));                              "
"    p[1] = read_imagef (input, sampler, (int2)(x0 + i, y0 + j + 1));                          "
"    ++i;                                                                                      "
"    p[2] = read_imagef (input, sampler, (int2)(x0 + i, y0 + j));                              "
"    p[3] = read_imagef (input, sampler, (int2)(x0 + i, y0 + j + 1));                          "
"    sum_gr += p[0].x;                                                                         "
"    sum_b += p[1].x;                                                                          "
"    sum_r += p[2].x;                                                                          "
"    sum_gb += p[3].x;                                                                         "

// grid (3, 0)
"    ++i;                                                                                      "
"    p[0] = read_imagef (input, sampler, (int2)(x0 + i, y0 + j));                              "
"    p[1] = read_imagef (input, sampler, (int2)(x0 + i, y0 + j + 1));                          "
"    ++i;                                                                                      "
"    p[2] = read_imagef (input, sampler, (int2)(x0 + i, y0 + j));                              "
"    p[3] = read_imagef (input, sampler, (int2)(x0 + i, y0 + j + 1));                          "
"    sum_gr += p[0].x;                                                                         "
"    sum_b += p[1].x;                                                                          "
"    sum_r += p[2].x;                                                                          "
"    sum_gb += p[3].x;                                                                         "

// grid (4, 0)
"    ++i;                                                                                      "
"    p[0] = read_imagef (input, sampler, (int2)(x0 + i, y0 + j));                              "
"    p[1] = read_imagef (input, sampler, (int2)(x0 + i, y0 + j + 1));                          "
"    ++i;                                                                                      "
"    p[2] = read_imagef (input, sampler, (int2)(x0 + i, y0 + j));                              "
"    p[3] = read_imagef (input, sampler, (int2)(x0 + i, y0 + j + 1));                          "
"    sum_gr += p[0].x;                                                                         "
"    sum_b += p[1].x;                                                                          "
"    sum_r += p[2].x;                                                                          "
"    sum_gb += p[3].x;                                                                         "

// grid (5, 0)
"    ++i;                                                                                      "
"    p[0] = read_imagef (input, sampler, (int2)(x0 + i, y0 + j));                              "
"    p[1] = read_imagef (input, sampler, (int2)(x0 + i, y0 + j + 1));                          "
"    ++i;                                                                                      "
"    p[2] = read_imagef (input, sampler, (int2)(x0 + i, y0 + j));                              "
"    p[3] = read_imagef (input, sampler, (int2)(x0 + i, y0 + j + 1));                          "
"    sum_gr += p[0].x;                                                                         "
"    sum_b += p[1].x;                                                                          "
"    sum_r += p[2].x;                                                                          "
"    sum_gb += p[3].x;                                                                         "

// grid (6, 0)
"    ++i;                                                                                      "
"    p[0] = read_imagef (input, sampler, (int2)(x0 + i, y0 + j));                              "
"    p[1] = read_imagef (input, sampler, (int2)(x0 + i, y0 + j + 1));                          "
"    ++i;                                                                                      "
"    p[2] = read_imagef (input, sampler, (int2)(x0 + i, y0 + j));                              "
"    p[3] = read_imagef (input, sampler, (int2)(x0 + i, y0 + j + 1));                          "
"    sum_gr += p[0].x;                                                                         "
"    sum_b += p[1].x;                                                                          "
"    sum_r += p[2].x;                                                                          "
"    sum_gb += p[3].x;                                                                         "


// grid (7, 0)
"    ++i;                                                                                      "
"    p[0] = read_imagef (input, sampler, (int2)(x0 + i, y0 + j));                              "
"    p[1] = read_imagef (input, sampler, (int2)(x0 + i, y0 + j + 1));                          "
"    ++i; "
"    p[2] = read_imagef (input, sampler, (int2)(x0 + i, y0 + j));                              "
"    p[3] = read_imagef (input, sampler, (int2)(x0 + i, y0 + j + 1));                          "
"    sum_gr += p[0].x;                                                                         "
"    sum_b += p[1].x;                                                                          "
"    sum_r += p[2].x;                                                                          "
"    sum_gb += p[3].x;                                                                         "

//end for loop
"    }                                                                                         "

"   avg_gr = sum_gr/count;                                                                     "
"   avg_r = sum_r/count;                                                                       "
"   avg_b = sum_b/count;                                                                       "
"   avg_gb = sum_gb/count;                                                                     "

"   output[y * w + x].avg_gr = convert_uint(avg_gr * 256.0);                                   "
"   output[y * w + x].avg_r = convert_uint(avg_r * 256.0);                                     "
"   output[y * w + x].avg_b = convert_uint(avg_b * 256.0);                                     "
"   output[y * w + x].avg_gb = convert_uint(avg_gb * 256.0);                                   "
"   output[y * w + x].valid_wb_count = 255;                                                    "
"   output[y * w + x].avg_y = convert_uint(((avg_gr + avg_gb)/2.0f)*256.0);                    "
"   output[y * w + x].f_value1 = 0;                                                            "
"   output[y * w + x].f_value2 = 0;                                                            "

"}                                                                                             "

