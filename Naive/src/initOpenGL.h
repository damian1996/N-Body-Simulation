#ifndef INITOPENGL_H
#define INITOPENGL_H

#include <cstdio>
#include "gl.h"
#include <stdexcept>
#include "render.h"
#include <unistd.h>

class initOpenGL {
    GLFWwindow* window;
    int licznik;
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
