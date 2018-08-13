#version 310 es

layout (local_size_x = 8, local_size_y = 8) in;

layout (binding = 0) readonly buffer InBuf {
    uvec4 data[];
} in_buf;

layout (binding = 1) writeonly buffer OutBuf {
    uvec4 data[];
} out_buf;

uniform uint in_img_width;
uniform uint in_x_offset;

uniform uint out_img_width;
uniform uint out_x_offset;

uniform uint copy_width;

void main ()
{
    uint g_x = gl_GlobalInvocationID.x;
    uint g_y = gl_GlobalInvocationID.y;

    if (g_x >= copy_width)
        return;

    out_buf.data[g_y * out_img_width + out_x_offset + g_x] = in_buf.data[g_y * in_img_width + in_x_offset + g_x];
}
