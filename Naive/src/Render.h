#ifndef RENDER_H
#define RENDER_H

#include <cstdio>
#include <stdexcept>
#include <thrust/host_vector.h>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <glm/vec3.hpp> // glm::vec3
#include <glm/vec4.hpp> // glm::vec4
#include <glm/mat4x4.hpp> // glm::mat4
#include <glm/gtc/matrix_transform.hpp> // glm::translate, glm::rotate, glm::scale, glm::perspective

#include "gl.h"
#include "RandomGenerators.h"

typedef thrust::host_vector<float > tf3;

class Render {
public:
    Render(std::vector<float> masses, unsigned numberOfBodies);
    ~Render();
    void createAndBindBuffer();
    void createAndCompileShaders();
    void createAndLinkProgram();
    void setupWindow();
    void init();
    void setupOpenGL();

    bool clearWindow();
    void destroyWindow();
    bool swapBuffers();

    static void mouse_pressed();
    static void mouse_released();
    static void mouse_scroll(double offset);
    static void mouse_move(double xpos, double ypos);

    static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
    static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
    static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos);

    void render();

    bool draw(thrust::host_vector<float>& positions);
    GLuint load_shader(const char* path, int shader_type);
    float getTime();
private:
    static float camera_theta;
    static float camera_phi;
    static float camera_radius;

    static bool is_mouse_pressed;
    static double mouse_position_x;
    static double mouse_position_y;

    const int width = 1000;
    const int height = 1000;
    float last_time;

    char* colorsToRender;
    float* positionsToRender;
    float* massesToRender;

    GLuint buffer[3];
    GLuint program, shFragment, shVertex;
    GLFWwindow* window;

    RandomGenerators* randomGenerator;
    unsigned numberOfBodies;
};

#endif
