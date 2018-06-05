#include "initOpenGL.h"

initOpenGL::initOpenGL() {
    licznik = 0;
}

initOpenGL::~initOpenGL() {
}

void error_callback(int error, const char* description) {
  throw std::runtime_error(description);
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    glfwSetWindowShouldClose(window, 1);
}

void initOpenGL::init() {
    if (!glfwInit())
        throw std::runtime_error("Failed to initialize glfw");
    glfwSetErrorCallback(error_callback);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    this->window = glfwCreateWindow(1000, 1000, "N-BODY SIMULATION", NULL, NULL);
    if (!this->window)
        throw std::runtime_error("Failed to create window");
}

void initOpenGL::setupWindow() {
    glfwMakeContextCurrent(this->window);
    glfwSetKeyCallback(this->window, key_callback);
    glfwSwapInterval(1); //enables v-sync
}

bool initOpenGL::ClearWindow() {
    if(!glfwWindowShouldClose(this->window)) {
        int width, height;
        glfwGetFramebufferSize(this->window, &width, &height);
        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT);
        return false;
    }
    else {
        destroyWindow();
        //printf("%d \n ", licznik++);
        return true;
    }
}

bool initOpenGL::Swap() {
    if(!glfwWindowShouldClose(this->window)) {
        glfwSwapBuffers(this->window);
        glfwPollEvents();
        licznik++;
        return false;
    }
    else {
        destroyWindow();
        //printf("%d \n ", licznik++);
        return true;
    }
}

void initOpenGL::destroyWindow() {
    glfwDestroyWindow(this->window);
    glfwTerminate();
}
