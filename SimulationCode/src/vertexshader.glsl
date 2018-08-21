#version 130
in vec3 position;
in vec3 color;
in float mass;
out vec3 vertex_color;
out float vMass;
uniform mat4 MVP;

void main() {
  gl_Position = MVP * vec4(position, 1);
  if(mass > 100000.0) {
    gl_PointSize = 13.0;
  } else {
    gl_PointSize = (mass/20000.0) + 3.0;
  }
  vMass = mass;
  vertex_color = color;
}
