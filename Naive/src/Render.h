#ifndef RENDER_H
#define RENDER_H

#include <cstdio>
#include <fstream>
#include <glm/gtc/matrix_transform.hpp> // glm::translate, glm::rotate, glm::scale, glm::perspective
#include <glm/mat4x4.hpp> // glm::mat4
#include <glm/vec3.hpp>   // glm::vec3
#include <glm/vec4.hpp>   // glm::vec4
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <thrust/host_vector.h>
#include <unistd.h>

#include "RandomGenerators.h"
#include "gl.h"

typedef thrust::host_vector<double> tf3;
typedef thrust::host_vector<char> tb3;
typedef std::vector<std::array<double, 3>> f3;
typedef std::vector<std::array<char, 3>> b3;
typedef std::vector<double> w1;
typedef double double3d[3];
typedef unsigned char byte3d[3];

class initOpenGL;

class Render {
public:
  int counter = 0;
  Render(std::vector<double>& masses, unsigned N);
  ~Render();
  void createAndBindBuffer();
  void createAndCompileShaders();
  void createAndLinkProgram();
  void setupWindow();
  void init();
  void setupOpenGL();

  bool ClearWindow();
  void destroyWindow();
  bool Swap();

  static void mouse_pressed();
  static void mouse_released();
  static void mouse_scroll(double offset);
  static void mouse_move(double xpos, double ypos);

  static void key_callback(GLFWwindow *window, int key, int scancode,
                           int action, int mods);
  static void scroll_callback(GLFWwindow *window, double xoffset,
                              double yoffset);
  static void mouse_button_callback(GLFWwindow *window, int button, int action,
                                    int mods);
  static void cursor_position_callback(GLFWwindow *window, double xpos,
                                       double ypos);

  void render();

  bool draw(thrust::host_vector<double> &positions);
  GLuint load_shader(const char *path, int shader_type);
  double getTime();

private:
  static double camera_theta;
  static double camera_phi;
  static double camera_radius;

  static bool is_mouse_pressed;
  static double mouse_position_x;
  static double mouse_position_y;

  int width = 1000, height = 1000;
  double last_time;

  char *V_color;
  float *V_position;
  double *V_mass;

  GLuint buffer[3];
  GLuint program, sh_fragment, sh_vertex;
  GLFWwindow *window;

  RandomGenerators *rg;
  unsigned N;
};

#endif
