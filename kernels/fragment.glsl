#version 430

#define NB_COLORS 6

uniform sampler2D tex;

uniform float min;
uniform float max;

in vec2 texture_coord;

out vec4 fColor;

vec4 color_map[NB_COLORS+1] = {vec4(0.0f,   0.0f,   0.0f,   1.0f),
			       vec4(0.188f, 0.271f, 0.741f, 1.0f),
			       vec4(0.055f, 0.569f, 0.824,  1.0f),
			       vec4(0.251f, 0.729f, 0.600f, 1.0f),
			       vec4(0.937f, 0.725f, 0.294f, 1.0f),
			       vec4(0.976f, 0.984f, 0.051f, 1.0f),
			       vec4(0.976f, 0.984f, 0.051f, 1.0f)};

void main(void) {
  float normalized_value = (texture(tex, texture_coord).x-min)/(max-min);

  int idx = int(floor(normalized_value*NB_COLORS));

  float lambda = normalized_value*NB_COLORS-float(idx);

  fColor = lambda*color_map[idx+1]+(1.0f-lambda)*color_map[idx];
}
