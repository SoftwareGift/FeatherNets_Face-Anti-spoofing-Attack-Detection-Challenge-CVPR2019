/*
 * function: kernel_denoise
 *     bi-laterial filter for denoise usage
 * input:    image2d_t as read only
 * output:   image2d_t as write only
 * sigma_r:  the parameter to set sigma_r in the Gaussian filtering
 * imw:      image width, used for edge detect
 * imh:      image height, used for edge detect
 * vertical_offset: used for get the uv plane
 */

__constant float gausssingle[25] = {0.6411f, 0.7574f, 0.8007f, 0.7574f, 0.6411f, 0.7574f, 0.8948f, 0.9459f, 0.8948f, 0.7574f, 0.8007f, 0.94595945f, 1.0f, 0.9459f, 0.8007f, 0.7574f, 0.8948f, 0.9459f, 0.8948f, 0.7574f, 0.6411f, 0.7574f, 0.8007f, 0.7574f, 0.6411f};

#define LOCAL_SIZE_X 16
#define LOCAL_SIZE_Y 15

__kernel void kernel_biyuv(__read_only image2d_t srcYUV, __write_only image2d_t dstYUV, float sigma_r, unsigned int imw, unsigned int imh, uint vertical_offset )
{
    int x = get_global_id(1);    //[0,imw-1]
    int y = get_global_id(0);    //[0,imh-1]
    int localX = get_local_id(1);    //[0,imw/120-1]
    int localY = get_local_id(0);    //[0,imh/72-1]
    //printf("localX=%d,localY=%d\n",localX,localY);

    float normF = 0.0f;
    float H = 0.0f;
    float delta = 0.0f;
    int i = 0, j = 0;
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
    sigma_r = 2 * pown(sigma_r, 2);

    //coord in srcY
    float4 line;
    line.x = 0.0f;
    line.y = 0.0f;
    line.z = 0.0f;
    line.w = 1.0f;
    float4 uv_in;

    // cpy UV
    if(y % 2 == 0) {
        uv_in = read_imagef(srcYUV, sampler, (int2)(x, y / 2 + vertical_offset));
        write_imagef(dstYUV, (int2)(x, y / 2 + vertical_offset), uv_in);
    }


    __local float4 pixel[LOCAL_SIZE_X + 4][LOCAL_SIZE_Y + 4];
    bool interior = x > 1 && x < (imw - 3)
                    && y > 1 && y < (imh - 3);
    if(interior)
    {
        pixel[localX + 2][localY + 2] = read_imagef(srcYUV, sampler, (int2)(x, y));

        if(localX == 0)
        {
            if(localY == 0)
            {
                pixel[0][0] = read_imagef(srcYUV, sampler, (int2)(x - 2, y - 2));
                pixel[0][1] = read_imagef(srcYUV, sampler, (int2)(x - 2, y - 1));
                pixel[0][2] = read_imagef(srcYUV, sampler, (int2)(x - 2, y  ));
                pixel[1][0] = read_imagef(srcYUV, sampler, (int2)(x - 1, y - 2));
                pixel[1][1] = read_imagef(srcYUV, sampler, (int2)(x - 1, y - 1));
                pixel[1][2] = read_imagef(srcYUV, sampler, (int2)(x - 1, y  ));
                pixel[2][0] = read_imagef(srcYUV, sampler, (int2)(x  , y - 2));
                pixel[2][1] = read_imagef(srcYUV, sampler, (int2)(x  , y - 1));
            }
            else if(localY == LOCAL_SIZE_Y - 1)
            {
                pixel[0][LOCAL_SIZE_Y - 1 + 2] = read_imagef(srcYUV, sampler, (int2)(x - 2, y  ));
                pixel[0][LOCAL_SIZE_Y  + 2] = read_imagef(srcYUV, sampler, (int2)(x - 2, y + 1));
                pixel[0][LOCAL_SIZE_Y + 1 + 2] = read_imagef(srcYUV, sampler, (int2)(x - 2, y + 2));
                pixel[1][LOCAL_SIZE_Y - 1 + 2] = read_imagef(srcYUV, sampler, (int2)(x - 1, y  ));
                pixel[1][LOCAL_SIZE_Y  + 2] = read_imagef(srcYUV, sampler, (int2)(x - 1, y + 1));
                pixel[1][LOCAL_SIZE_Y + 1 + 2] = read_imagef(srcYUV, sampler, (int2)(x - 1, y + 2));
                pixel[2][LOCAL_SIZE_Y  + 2] = read_imagef(srcYUV, sampler, (int2)(x  , y + 1));
                pixel[2][LOCAL_SIZE_Y + 1 + 2] = read_imagef(srcYUV, sampler, (int2)(x  , y + 2));
            }
            else
            {
                pixel[0][localY + 2] = read_imagef(srcYUV, sampler, (int2)(x - 2, y));
                pixel[1][localY + 2] = read_imagef(srcYUV, sampler, (int2)(x - 1, y));
            }
        }
        else if(localX == LOCAL_SIZE_X - 1)
        {
            if(localY == 0)
            {
                pixel[LOCAL_SIZE_X - 1 + 2 + 0][0] = read_imagef(srcYUV, sampler, (int2)(x  , y - 2));
                pixel[LOCAL_SIZE_X - 1 + 2 + 0][1] = read_imagef(srcYUV, sampler, (int2)(x  , y - 1));
                //pixel[LOCAL_SIZE_X-1+2+0][2]=read_imagef(srcYUV, sampler,(int2)(x  ,y));
                pixel[LOCAL_SIZE_X - 1 + 2 + 1][0] = read_imagef(srcYUV, sampler, (int2)(x + 1, y - 2));
                pixel[LOCAL_SIZE_X - 1 + 2 + 1][1] = read_imagef(srcYUV, sampler, (int2)(x + 1, y - 1));
                pixel[LOCAL_SIZE_X - 1 + 2 + 1][2] = read_imagef(srcYUV, sampler, (int2)(x + 1, y  ));
                pixel[LOCAL_SIZE_X - 1 + 2 + 2][0] = read_imagef(srcYUV, sampler, (int2)(x + 2, y - 2));
                pixel[LOCAL_SIZE_X - 1 + 2 + 2][1] = read_imagef(srcYUV, sampler, (int2)(x + 2, y - 1));
                pixel[LOCAL_SIZE_X - 1 + 2 + 2][2] = read_imagef(srcYUV, sampler, (int2)(x + 2, y  ));
            }
            else if(localY == LOCAL_SIZE_Y - 1)
            {
                //  pixel[LOCAL_SIZE_X-1+2+0][LOCAL_SIZE_Y-1+2+0]=read_imagef(srcYUV, sampler,(int2)(x  ,y  ));
                pixel[LOCAL_SIZE_X - 1 + 2 + 0][LOCAL_SIZE_Y - 1 + 2 + 1] = read_imagef(srcYUV, sampler, (int2)(x  , y + 1));
                pixel[LOCAL_SIZE_X - 1 + 2 + 0][LOCAL_SIZE_Y - 1 + 2 + 2] = read_imagef(srcYUV, sampler, (int2)(x  , y + 2));
                pixel[LOCAL_SIZE_X - 1 + 2 + 1][LOCAL_SIZE_Y - 1 + 2 + 0] = read_imagef(srcYUV, sampler, (int2)(x + 1, y  ));
                pixel[LOCAL_SIZE_X - 1 + 2 + 1][LOCAL_SIZE_Y - 1 + 2 + 1] = read_imagef(srcYUV, sampler, (int2)(x + 1, y + 1));
                pixel[LOCAL_SIZE_X - 1 + 2 + 1][LOCAL_SIZE_Y - 1 + 2 + 2] = read_imagef(srcYUV, sampler, (int2)(x + 1, y + 2));
                pixel[LOCAL_SIZE_X - 1 + 2 + 2][LOCAL_SIZE_Y - 1 + 2 + 0] = read_imagef(srcYUV, sampler, (int2)(x + 2, y  ));
                pixel[LOCAL_SIZE_X - 1 + 2 + 2][LOCAL_SIZE_Y - 1 + 2 + 1] = read_imagef(srcYUV, sampler, (int2)(x + 2, y + 1));
                pixel[LOCAL_SIZE_X - 1 + 2 + 2][LOCAL_SIZE_Y - 1 + 2 + 2] = read_imagef(srcYUV, sampler, (int2)(x + 2, y + 2));
            }
            else
            {
                pixel[LOCAL_SIZE_X - 1 + 2 + 1][localY + 2] = read_imagef(srcYUV, sampler, (int2)(x + 1, y  ));
                pixel[LOCAL_SIZE_X - 1 + 2 + 2][localY + 2] = read_imagef(srcYUV, sampler, (int2)(x + 2, y  ));
            }
        }
        else if(localY == 0)
        {
            pixel[localX + 2][0] = read_imagef(srcYUV, sampler, (int2)(x, y - 2));
            pixel[localX + 2][1] = read_imagef(srcYUV, sampler, (int2)(x, y - 1));
        }
        else if(localY == LOCAL_SIZE_Y - 1)
        {
            pixel[localX + 2][LOCAL_SIZE_Y - 1 + 2 + 1] = read_imagef(srcYUV, sampler, (int2)(x, y + 1));
            pixel[localX + 2][LOCAL_SIZE_Y - 1 + 2 + 2] = read_imagef(srcYUV, sampler, (int2)(x, y + 2));
        }
    } else {
        line = read_imagef(srcYUV, sampler, (int2)(x, y));
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if (interior) {
#pragma unroll
        for(i = 0; i < 5; i++)
        {
#pragma unroll
            for(j = 0; j < 5; j++)
            {
                delta = pown(pixel[localX + i][localY + j].x - pixel[localX + 2][localY + 2].x, 2);
                H = (exp(-(delta / sigma_r))) * gausssingle[i * 5 + j];
                normF += H;
                line.x += pixel[localX + i][localY + j].x * H;
            }
        }

        line.x = line.x / normF;
    }

    write_imagef(dstYUV, (int2)(x, y), line);
}
