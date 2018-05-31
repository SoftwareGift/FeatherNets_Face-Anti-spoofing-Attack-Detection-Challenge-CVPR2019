/*
 * function: kernel_wire_frame
 *
 * output_y:              Y channel image2d_t as write only
 * output_uv:             uv channel image2d_t as write only
 * wire_frames_coords:    coordinates of wire frames
 * coords_num:            number of coordinates to be processed
 */

__kernel void kernel_wire_frame (
    __write_only image2d_t output_y, __write_only image2d_t output_uv,
    __global uint2 *wire_frames_coords, uint coords_num,
    float border_y, float border_u, float border_v)
{
    if (coords_num == 0) {
        return;
    }

    int gid = get_global_id (0);
    if (gid >= coords_num) {
        return;
    }

    uint2 coord = wire_frames_coords [gid];

    write_imagef (output_y, (int2)(coord.x / 2, coord.y), (float4)(border_y));
    if (coord.y % 2 == 0) {
        write_imagef (output_uv, (int2)(coord.x / 2, coord.y / 2), (float4)(border_u, border_v, 0.0f, 0.0f));
    }
}
