#ifndef RENDER_H
#define RENDER_H

#include <array>
#include <cstdio>
#include <stdexcept>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <vector>

#include "gl.h"
#include "initOpenGL.h"


typedef thrust::host_vector<float > tf3;
typedef thrust::host_vector<char > tb3;
typedef std::vector<std::array<float, 3> > f3;
typedef std::vector<std::array<char, 3> > b3;
typedef std::vector<float> w1;
typedef float float3d[3];
typedef unsigned char byte3d[3];

class initOpenGL;

class Scene {
  byte3d* V_color;
  float3d* V_position;
  float* V_weight;
  GLuint buffer[2];
  GLuint program, sh_fragment, sh_vertex;
  initOpenGL* opengl;
  unsigned N;
public:
  Scene(unsigned N);
  ~Scene();
  void createAndBindBuffer();
  void createAndCompileShaders();
  void createAndLinkProgram();
  void init();
  void render();
  void setupOpenGL();
  bool draw();
  void sendInitialData(tf3 positions, b3 colors, tf3 weights);
  void sendData(tf3 positions);
};

#endif
