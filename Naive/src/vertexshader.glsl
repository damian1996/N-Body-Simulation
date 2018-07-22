#version 130
in vec3 position;
in vec3 color;
in float mass;
out vec3 vertex_color;
out float vMass;
uniform mat4 MVP;

void main() {
  gl_Position = MVP * vec4(position, 1);
  vMass = mass;
  vertex_color = color;
}