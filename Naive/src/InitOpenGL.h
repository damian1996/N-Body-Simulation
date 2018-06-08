#ifndef INITOPENGL_H
#define INITOPENGL_H

#include <cstdio>
#include <stdexcept>
#include <unistd.h>

#include "gl.h"

class initOpenGL {
    GLFWwindow* window;
public:
    initOpenGL();
    ~initOpenGL();
    bool ClearWindow();
    void destroyWindow();
    void init();
    void setupWindow();
    bool Swap();
};

#endif
