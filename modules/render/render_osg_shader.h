/*
 * render_osg_shader.h -  common gl shaders
 *
 *  Copyright (c) 2018 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Author: Zong Wei <wei.zong@intel.com>
 */

#ifndef XCAM_OSG_SHADER_H
#define XCAM_OSG_SHADER_H

namespace XCam {

static const char VtxShaderProjectNV12Texture[] = ""
        "precision highp float;                                        \n"
        "attribute vec4 osg_Vertex;                                    \n"
        "attribute vec2 osg_MultiTexCoord0;                            \n"
        "attribute vec4 osg_Color;                                     \n"
        "uniform mat4 osg_ModelViewProjectionMatrix;                   \n"
        "uniform mat4 osg_ModelViewMatrix;                             \n"
        "uniform mat4 osg_ViewMatrixInverse;                           \n"
        "varying vec2 texcoord;                                        \n"
        "varying vec4 color;                                           \n"
        "void main(void)                                               \n"
        "{                                                             \n"
        "    texcoord = osg_MultiTexCoord0;                            \n"
        "    gl_Position = osg_ModelViewProjectionMatrix * osg_Vertex; \n"
        "    color = osg_Color;                                        \n"
        "}                                                             \n";

static const char FrgShaderProjectNV12Texture[] = ""
        "precision highp float;                                                      \n"
        "uniform sampler2D textureY;                                                 \n"
        "uniform sampler2D textureUV;                                                \n"
        "varying vec2 texcoord;                                                      \n"
        "varying vec4 color;                                                         \n"
        "vec4 getRGBColorNV12(sampler2D u_textureY, sampler2D u_textureUV, vec2 tex) \n"
        "{                                                                           \n"
        "    vec4 resultcolor = vec4 (0.0, 0.0, 0.0, 1.0);                           \n"
        "    float y, u, v;                                                          \n"
        "    y = texture2D(u_textureY, vec2(tex.s, tex.t)).r;                        \n"
        "    vec2 colorUV = texture2D(u_textureUV, vec2(tex.s, tex.t)).rg;            \n"
        "    u = colorUV.x-0.5;                                                      \n"
        "    v = colorUV.y-0.5;                                                      \n"
        "    y = 1.1643*(y-0.0625);                                                  \n"
        "    resultcolor.r = (y+1.5958*(v));                                         \n"
        "    resultcolor.g = (y-0.39173*(u)-0.81290*(v));                            \n"
        "    resultcolor.b = (y+2.017*(u));                                          \n"
        "    resultcolor.a = 1.0;                                                    \n"
        "    return resultcolor;                                                     \n"
        "}                                                                           \n"
        "void main()                                                                 \n"
        "{                                                                           \n"
        "    vec4 textureColor = getRGBColorNV12(textureY, textureUV, texcoord);     \n"
        "    gl_FragColor =  textureColor;                                           \n"
        "}                                                                           \n";

const char VtxShaderSimpleTexture[] = ""
                                      "precision highp float;                                        \n"
                                      "attribute vec4 osg_Vertex;                                    \n"
                                      "attribute vec4 osg_Color;                                     \n"
                                      "uniform mat4 osg_ModelViewProjectionMatrix;                   \n"
                                      "varying vec4 color;                                           \n"
                                      "void main(void)                                               \n"
                                      "{                                                             \n"
                                      "    gl_Position = osg_ModelViewProjectionMatrix * osg_Vertex; \n"
                                      "    color = osg_Color;                                        \n"
                                      "}                                                             \n";

const char FrgShaderSimpleTexture[] = ""
                                      "precision highp float;            \n"
                                      "varying vec4 color;               \n"
                                      "void main()                       \n"
                                      "{                                 \n"
                                      "    gl_FragColor = color;         \n"
                                      "}                                 \n";


} // namespace XCam

#endif // XCAM_OSG_SHADER_H
