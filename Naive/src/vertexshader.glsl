#version 130
in vec3 position;
in vec3 color;
in float mass;
out vec3 vertex_color;
out float vMass;
uniform mat4 MVP;

void main() {
  gl_Position = MVP * vec4(position, 1);
  if(mass < 1000.0) {
    gl_PointSize = 2.0;
  } else if((mass >= 1000.0f) && (mass <= 10000.0)) {
    gl_PointSize = 3.0;
  } else if((mass >= 10000.0) && (mass <= 50000.0)) {
    gl_PointSize = 5.0;
  } else {
    gl_PointSize = 7.0;
  }
  vMass = mass;
  vertex_color = color;
}
