#version 430

in vec4 vPosition;
in vec2 vTextureCoord;

out vec2 texture_coord;

void main(void){
    gl_Position = vPosition;
    texture_coord = vTextureCoord;
}
