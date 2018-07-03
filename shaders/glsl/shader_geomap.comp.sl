#version 310 es

layout (local_size_x = 8, local_size_y = 8) in;

layout (binding = 0) readonly buffer InBufY {
    uint data[];
} in_buf_y;

layout (binding = 1) readonly buffer InBufUV {
    uint data[];
} in_buf_uv;

layout (binding = 2) writeonly buffer OutBufY {
    uint data[];
} out_buf_y;

layout (binding = 3) writeonly buffer OutBufUV {
    uint data[];
} out_buf_uv;

layout (binding = 4) readonly buffer GeoMapTable {
    vec2 data[];
} lut;

uniform uint in_img_width;
uniform uint in_img_height;

uniform uint out_img_width;
uniform uint out_img_height;

uniform uint lut_width;
uniform uint lut_height;

uniform vec4 lut_step;
uniform vec2 lut_std_step;

#define UNIT_SIZE 4u

#define unpack_unorm_y(index) \
    { \
        vec4 value = unpackUnorm4x8 (in_buf_y.data[index00[index]]); \
        out_y00[index] = value[x00_fract[index]]; \
        value = unpackUnorm4x8 (in_buf_y.data[index01[index]]); \
        out_y01[index] = value[x01_fract[index]]; \
        value = unpackUnorm4x8 (in_buf_y.data[index10[index]]); \
        out_y10[index] = value[x10_fract[index]]; \
        value = unpackUnorm4x8 (in_buf_y.data[index11[index]]); \
        out_y11[index] = value[x11_fract[index]]; \
    }

void geomap_y (vec4 lut_x, vec4 lut_y, out vec4 in_img_x, out vec4 in_img_y, out bvec4 out_bound, out uint out_data);
void geomap_uv (vec2 in_uv_x, vec2 in_uv_y, bvec4 out_bound_uv, out uint out_data);

void main ()
{
    uint g_x = gl_GlobalInvocationID.x;
    uint g_y = gl_GlobalInvocationID.y * 2u;

    vec2 cent = (vec2 (out_img_width, out_img_height) - 1.0f) / 2.0f;
    vec2 step = g_x < uint (cent.x) ? lut_step.xy : lut_step.zw;

    vec2 start = (vec2 (g_x, g_y) - cent) * step + cent * lut_std_step;
    vec4 lut_x = start.x * float (UNIT_SIZE) + vec4 (0.0f, step.x, step.x * 2.0f, step.x * 3.0f);
    vec4 lut_y = start.yyyy;
    lut_x = clamp (lut_x, 0.0f, float (lut_width) - 1.0f);
    lut_y = clamp (lut_y, 0.0f, float (lut_height) - 1.0f - step.y);

    uint out_data;
    vec4 in_img_x, in_img_y;
    bvec4 out_bound;
    geomap_y (lut_x, lut_y, in_img_x, in_img_y, out_bound, out_data);
    out_buf_y.data[g_y * out_img_width + g_x] = out_data;

    bvec4 out_bound_uv = out_bound.xxzz;
    if (all (out_bound_uv)) {
        out_data = packUnorm4x8 (vec4 (0.5f));
    } else {
        vec2 in_uv_x = in_img_x.xz;
        vec2 in_uv_y = in_img_y.xz / 2.0f;
        geomap_uv (in_uv_x, in_uv_y, out_bound_uv, out_data);
    }
    out_buf_uv.data[g_y / 2u * out_img_width + g_x] = out_data;

    lut_y += step.y;
    geomap_y (lut_x, lut_y, in_img_x, in_img_y, out_bound, out_data);
    out_buf_y.data[(g_y + 1u) * out_img_width + g_x] = out_data;
}

void geomap_y (vec4 lut_x, vec4 lut_y, out vec4 in_img_x, out vec4 in_img_y, out bvec4 out_bound, out uint out_data)
{
    uvec4 x00 = uvec4 (lut_x);
    uvec4 y00 = uvec4 (lut_y);
    uvec4 x01 = x00 + 1u;
    uvec4 y01 = y00;
    uvec4 x10 = x00;
    uvec4 y10 = y00 + 1u;
    uvec4 x11 = x01;
    uvec4 y11 = y10;

    vec4 fract_x = fract (lut_x);
    vec4 fract_y = fract (lut_y);
    vec4 weight00 = (1.0f - fract_x) * (1.0f - fract_y);
    vec4 weight01 = fract_x * (1.0f - fract_y);
    vec4 weight10 = (1.0f - fract_x) * fract_y;
    vec4 weight11 = fract_x * fract_y;

    uvec4 index00 = y00 * lut_width + x00;
    uvec4 index01 = y01 * lut_width + x01;
    uvec4 index10 = y10 * lut_width + x10;
    uvec4 index11 = y11 * lut_width + x11;

    vec4 in_img_x00, in_img_x01, in_img_x10, in_img_x11;
    vec4 in_img_y00, in_img_y01, in_img_y10, in_img_y11;
    for (uint i = 0u; i < UNIT_SIZE; ++i) {
        vec2 value = lut.data[index00[i]];
        in_img_x00[i] = value.x;
        in_img_y00[i] = value.y;
        value = lut.data[index01[i]];
        in_img_x01[i] = value.x;
        in_img_y01[i] = value.y;
        value = lut.data[index10[i]];
        in_img_x10[i] = value.x;
        in_img_y10[i] = value.y;
        value = lut.data[index11[i]];
        in_img_x11[i] = value.x;
        in_img_y11[i] = value.y;
    }
    in_img_x = in_img_x00 * weight00 + in_img_x01 * weight01 + in_img_x10 * weight10 + in_img_x11 * weight11;
    in_img_y = in_img_y00 * weight00 + in_img_y01 * weight01 + in_img_y10 * weight10 + in_img_y11 * weight11;

    for (uint i = 0u; i < UNIT_SIZE; ++i) {
        out_bound[i] = in_img_x[i] < 0.0f || in_img_x[i] > float (in_img_width * UNIT_SIZE - 1u) ||
                       in_img_y[i] < 0.0f || in_img_y[i] > float (in_img_height - 1u);
    }
    if (all (out_bound)) {
        out_data = 0u;
        return;
    }

    x00 = uvec4 (in_img_x);
    y00 = uvec4 (in_img_y);
    x01 = x00 + 1u;
    y01 = y00;
    x10 = x00;
    y10 = y00 + 1u;
    x11 = x01;
    y11 = y10;

    fract_x = fract (in_img_x);
    fract_y = fract (in_img_y);
    weight00 = (1.0f - fract_x) * (1.0f - fract_y);
    weight01 = fract_x * (1.0f - fract_y);
    weight10 = (1.0f - fract_x) * fract_y;
    weight11 = fract_x * fract_y;

    uvec4 x00_floor = x00 / UNIT_SIZE;
    uvec4 x01_floor = x01 / UNIT_SIZE;
    uvec4 x10_floor = x10 / UNIT_SIZE;
    uvec4 x11_floor = x11 / UNIT_SIZE;
    uvec4 x00_fract = x00 % UNIT_SIZE;
    uvec4 x01_fract = x01 % UNIT_SIZE;
    uvec4 x10_fract = x10 % UNIT_SIZE;
    uvec4 x11_fract = x11 % UNIT_SIZE;

    index00 = y00 * in_img_width + x00_floor;
    index01 = y01 * in_img_width + x01_floor;
    index10 = y10 * in_img_width + x10_floor;
    index11 = y11 * in_img_width + x11_floor;

    // pixel Y-value
    vec4 out_y00, out_y01, out_y10, out_y11;
    unpack_unorm_y (0);
    unpack_unorm_y (1);
    unpack_unorm_y (2);
    unpack_unorm_y (3);

    vec4 inter_y = out_y00 * weight00 + out_y01 * weight01 + out_y10 * weight10 + out_y11 * weight11;
    out_data = packUnorm4x8 (inter_y * vec4 (not (out_bound)));
}

void geomap_uv (vec2 in_uv_x, vec2 in_uv_y, bvec4 out_bound_uv, out uint out_data)
{
    uvec2 x00 = uvec2 (in_uv_x);
    uvec2 y00 = uvec2 (in_uv_y);
    uvec2 x01 = x00 + 1u;
    uvec2 y01 = y00;
    uvec2 x10 = x00;
    uvec2 y10 = y00 + 1u;
    uvec2 x11 = x01;
    uvec2 y11 = y10;

    vec2 fract_x = fract (in_uv_x);
    vec2 fract_y = fract (in_uv_y);
    vec2 weight00 = (1.0f - fract_x) * (1.0f - fract_y);
    vec2 weight01 = fract_x * (1.0f - fract_y);
    vec2 weight10 = (1.0f - fract_x) * fract_y;
    vec2 weight11 = fract_x * fract_y;

    uvec2 x00_floor = x00 / UNIT_SIZE;
    uvec2 x01_floor = x01 / UNIT_SIZE;
    uvec2 x10_floor = x10 / UNIT_SIZE;
    uvec2 x11_floor = x11 / UNIT_SIZE;
    uvec2 x00_fract = (x00 % UNIT_SIZE) / 2u;
    uvec2 x01_fract = (x01 % UNIT_SIZE) / 2u;
    uvec2 x10_fract = (x10 % UNIT_SIZE) / 2u;
    uvec2 x11_fract = (x11 % UNIT_SIZE) / 2u;

    uvec2 index00 = y00 * in_img_width + x00_floor;
    uvec2 index01 = y01 * in_img_width + x01_floor;
    uvec2 index10 = y10 * in_img_width + x10_floor;
    uvec2 index11 = y11 * in_img_width + x11_floor;

    // pixel UV-value
    vec4 out_uv00, out_uv01, out_uv10, out_uv11;
    vec4 value = unpackUnorm4x8 (in_buf_uv.data[index00.x]);
    out_uv00.xy = x00_fract.x == 0u ? value.xy : value.zw;
    value = unpackUnorm4x8 (in_buf_uv.data[index01.x]);
    out_uv01.xy = x01_fract.x == 0u ? value.xy : value.zw;
    value = unpackUnorm4x8 (in_buf_uv.data[index10.x]);
    out_uv10.xy = x10_fract.x == 0u ? value.xy : value.zw;
    value = unpackUnorm4x8 (in_buf_uv.data[index11.x]);
    out_uv11.xy = x11_fract.x == 0u ? value.xy : value.zw;

    value = unpackUnorm4x8 (in_buf_uv.data[index00.y]);
    out_uv00.zw = x00_fract.y == 0u ? value.xy : value.zw;
    value = unpackUnorm4x8 (in_buf_uv.data[index01.y]);
    out_uv01.zw = x01_fract.y == 0u ? value.xy : value.zw;
    value = unpackUnorm4x8 (in_buf_uv.data[index10.y]);
    out_uv10.zw = x10_fract.y == 0u ? value.xy : value.zw;
    value = unpackUnorm4x8 (in_buf_uv.data[index11.y]);
    out_uv11.zw = x11_fract.y == 0u ? value.xy : value.zw;

    vec4 inter_uv = out_uv00 * weight00.xxyy + out_uv01 * weight01.xxyy +
                    out_uv10 * weight10.xxyy + out_uv11 * weight11.xxyy;
    inter_uv = inter_uv * vec4 (not (out_bound_uv)) + vec4 (out_bound_uv) * 0.5f;
    out_data = packUnorm4x8 (inter_uv);
}
