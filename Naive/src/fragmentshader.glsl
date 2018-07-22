#version 130
uniform mat4 MVP;
in float vMass;
in vec3 vertex_color;
void main() {
  gl_FragColor = vec4(vertex_color, 0.5);
}
