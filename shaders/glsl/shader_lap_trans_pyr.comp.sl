#version 310 es

layout (local_size_x = 8, local_size_y = 8) in;

layout (binding = 0) readonly buffer InBufY {
    uvec2 data[];
} in_buf_y;

layout (binding = 1) readonly buffer InBufUV {
    uvec2 data[];
} in_buf_uv;

layout (binding = 2) readonly buffer GaussScaleBufY {
    uint data[];
} gaussscale_buf_y;

layout (binding = 3) readonly buffer GaussScaleBufUV {
    uint data[];
} gaussscale_buf_uv;

layout (binding = 4) writeonly buffer OutBufY {
    uvec2 data[];
} out_buf_y;

layout (binding = 5) writeonly buffer OutBufUV {
    uvec2 data[];
} out_buf_uv;

uniform uint in_img_width;
uniform uint in_img_height;
uniform uint in_offset_x;

uniform uint gaussscale_img_width;
uniform uint gaussscale_img_height;

uniform uint merge_width;

// normalization of half gray level
const float norm_half_gl = 128.0f / 255.0f;

void lap_trans_y (uvec2 y_id, uvec2 gs_id);
void lap_trans_uv (uvec2 uv_id, uvec2 gs_id);

void main ()
{
    uvec2 g_id = gl_GlobalInvocationID.xy;

    uvec2 y_id = uvec2 (g_id.x, g_id.y * 4u);
    y_id.x = clamp (y_id.x, 0u, merge_width - 1u);

    uvec2 gs_id = uvec2 (g_id.x, g_id.y * 2u);
    gs_id.x = clamp (gs_id.x, 0u, gaussscale_img_width - 1u);
    lap_trans_y (y_id, gs_id);

    y_id.y += 2u;
    gs_id.y += 1u;
    lap_trans_y (y_id, gs_id);

    uvec2 uv_id = uvec2 (y_id.x, g_id.y * 2u);
    gs_id.y = g_id.y;
    lap_trans_uv (uv_id, gs_id);
}

void lap_trans_y (uvec2 y_id, uvec2 gs_id)
{
    y_id.y = clamp (y_id.y, 0u, in_img_height - 1u);
    gs_id.y = clamp (gs_id.y, 0u, gaussscale_img_height - 1u);

    uint y_idx = y_id.y * in_img_width + in_offset_x + y_id.x;
    uvec2 in_pack = in_buf_y.data[y_idx];
    vec4 in0 = unpackUnorm4x8 (in_pack.x);
    vec4 in1 = unpackUnorm4x8 (in_pack.y);

    uint gs_idx = gs_id.y * gaussscale_img_width + gs_id.x;
    vec4 gs0 = unpackUnorm4x8 (gaussscale_buf_y.data[gs_idx]);
    vec4 gs1 = unpackUnorm4x8 (gaussscale_buf_y.data[gs_idx + 1u]);
    gs1 = (gs_id.x == gaussscale_img_width - 1u) ? gs0.wwww : gs1;

    vec4 inter = (gs0 + vec4 (gs0.yzw, gs1.x)) * 0.5f;
    vec4 inter00 = vec4 (gs0.x, inter.x, gs0.y, inter.y);
    vec4 inter01 = vec4 (gs0.z, inter.z, gs0.w, inter.w);

    vec4 lap0 = (in0 - inter00) * 0.5f + norm_half_gl;
    vec4 lap1 = (in1 - inter01) * 0.5f + norm_half_gl;
    lap0 = clamp (lap0, 0.0f, 1.0f);
    lap1 = clamp (lap1, 0.0f, 1.0f);

    uint out_idx = y_id.y * merge_width + y_id.x;
    out_buf_y.data[out_idx] = uvec2 (packUnorm4x8 (lap0), packUnorm4x8 (lap1));

    y_idx = (y_id.y >= in_img_height - 1u) ? y_idx : y_idx + in_img_width;
    in_pack = in_buf_y.data[y_idx];
    in0 = unpackUnorm4x8 (in_pack.x);
    in1 = unpackUnorm4x8 (in_pack.y);

    gs_idx = (gs_id.y >= gaussscale_img_height - 1u) ? gs_idx : gs_idx + gaussscale_img_width;
    gs0 = unpackUnorm4x8 (gaussscale_buf_y.data[gs_idx]);
    gs1 = unpackUnorm4x8 (gaussscale_buf_y.data[gs_idx + 1u]);
    gs1 = (gs_id.x == gaussscale_img_width - 1u) ? gs0.wwww : gs1;

    inter = (gs0 + vec4 (gs0.yzw, gs1.x)) * 0.5f;
    vec4 inter10 = (inter00 + vec4 (gs0.x, inter.x, gs0.y, inter.y)) * 0.5f;
    vec4 inter11 = (inter01 + vec4 (gs0.z, inter.z, gs0.w, inter.w)) * 0.5f;

    lap0 = (in0 - inter10) * 0.5f + norm_half_gl;
    lap1 = (in1 - inter11) * 0.5f + norm_half_gl;
    lap0 = clamp (lap0, 0.0f, 1.0f);
    lap1 = clamp (lap1, 0.0f, 1.0f);

    out_idx += merge_width;
    out_buf_y.data[out_idx] = uvec2 (packUnorm4x8 (lap0), packUnorm4x8 (lap1));
}

void lap_trans_uv (uvec2 uv_id, uvec2 gs_id)
{
    uv_id.y = clamp (uv_id.y, 0u, in_img_height / 2u - 1u);
    gs_id.y = clamp (gs_id.y, 0u, gaussscale_img_height / 2u - 1u);

    uint uv_idx = uv_id.y * in_img_width + in_offset_x + uv_id.x;
    uvec2 in_pack = in_buf_uv.data[uv_idx];
    vec4 in0 = unpackUnorm4x8 (in_pack.x);
    vec4 in1 = unpackUnorm4x8 (in_pack.y);

    uint gs_idx = gs_id.y * gaussscale_img_width + gs_id.x;
    vec4 gs0 = unpackUnorm4x8 (gaussscale_buf_uv.data[gs_idx]);
    vec4 gs1 = unpackUnorm4x8 (gaussscale_buf_uv.data[gs_idx + 1u]);
    gs1 = (gs_id.x == gaussscale_img_width - 1u) ? gs0.zwzw : gs1;

    vec4 inter = (gs0 + vec4 (gs0.zw, gs1.xy)) * 0.5f;
    vec4 inter00 = vec4 (gs0.xy, inter.xy);
    vec4 inter01 = vec4 (gs0.zw, inter.zw);

    vec4 lap0 = (in0 - inter00) * 0.5f + norm_half_gl;
    vec4 lap1 = (in1 - inter01) * 0.5f + norm_half_gl;
    lap0 = clamp (lap0, 0.0f, 1.0f);
    lap1 = clamp (lap1, 0.0f, 1.0f);

    uint out_idx = uv_id.y * merge_width + uv_id.x;
    out_buf_uv.data[out_idx] = uvec2 (packUnorm4x8 (lap0), packUnorm4x8 (lap1));

    uv_idx = (uv_id.y >= (in_img_height / 2u - 1u)) ? uv_idx : uv_idx + in_img_width;
    in_pack = in_buf_uv.data[uv_idx];
    in0 = unpackUnorm4x8 (in_pack.x);
    in1 = unpackUnorm4x8 (in_pack.y);

    gs_idx = (gs_id.y >= (gaussscale_img_height / 2u - 1u)) ? gs_idx : gs_idx + gaussscale_img_width;
    gs0 = unpackUnorm4x8 (gaussscale_buf_uv.data[gs_idx]);
    gs1 = unpackUnorm4x8 (gaussscale_buf_uv.data[gs_idx + 1u]);
    gs1 = (gs_id.x == gaussscale_img_width - 1u) ? gs0.zwzw : gs1;

    inter = (gs0 + vec4 (gs0.zw, gs1.xy)) * 0.5f;
    vec4 inter10 = (inter00 + vec4 (gs0.xy, inter.xy)) * 0.5f;
    vec4 inter11 = (inter01 + vec4 (gs0.zw, inter.zw)) * 0.5f;

    lap0 = (in0 - inter10) * 0.5f + norm_half_gl;
    lap1 = (in1 - inter11) * 0.5f + norm_half_gl;
    lap0 = clamp (lap0, 0.0f, 1.0f);
    lap1 = clamp (lap1, 0.0f, 1.0f);

    out_idx += merge_width;
    out_buf_uv.data[out_idx] = uvec2 (packUnorm4x8 (lap0), packUnorm4x8 (lap1));
}
