#ifndef RENDER_H
#define RENDER_H

#include <cstdio>
#include <stdexcept>
#include <thrust/host_vector.h>

#include "gl.h"
#include "InitOpenGL.h"
#include "RandomGenerators.h"

typedef thrust::host_vector<float > tf3;
typedef thrust::host_vector<char > tb3;
typedef std::vector<std::array<float, 3> > f3;
typedef std::vector<std::array<char, 3> > b3;
typedef std::vector<float> w1;
typedef float float3d[3];
typedef unsigned char byte3d[3];

class initOpenGL;

class Render {
  byte3d* V_color;
  double last_time;
  float3d* V_position;
  GLuint buffer[2];
  GLuint program, sh_fragment, sh_vertex;
  initOpenGL* opengl;
  RandomGenerators* rg;
  unsigned N;
public:
  int counter = 0;
  Render(unsigned N);
  ~Render();
  void createAndBindBuffer();
  void createAndCompileShaders();
  void createAndLinkProgram();
  void init();
  void render();
  void setupOpenGL();
  bool draw(thrust::host_vector<float>& positions);
  float getTime();
};

#endif
