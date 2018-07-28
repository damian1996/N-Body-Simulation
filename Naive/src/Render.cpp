#include "Render.h"
#include <iostream>

float Render::camera_theta = 0.02;
float Render::camera_phi = 0.01;
float Render::camera_radius = 4;
bool Render::is_mouse_pressed = false;
double Render::mouse_position_x = 0;
double Render::mouse_position_y = 0;

Render::Render(std::vector<float> masses, unsigned numberOfBodies) : numberOfBodies(numberOfBodies) {
    positionsToRender = new float[3*numberOfBodies];
    colorsToRender = new char[3*numberOfBodies];
    massesToRender = new float[numberOfBodies];
    for(unsigned i=0; i<numberOfBodies; i++) {
        massesToRender[i] = masses[i];
    }
    randomGenerator = new RandomGenerators();
    for(unsigned i=0; i<numberOfBodies; i++) {
       for(unsigned j=0; j<3; j++) {
          colorsToRender[i*3 + j] = randomGenerator->getRandomColor();
       }
    }
}

Render::~Render() {
    delete randomGenerator;
    delete [] massesToRender;
    delete [] colorsToRender;
    delete [] positionsToRender;
}

void Render::mouse_move(double xpos, double ypos) {
  if(is_mouse_pressed) {
    camera_phi += (ypos - mouse_position_y)*0.005;
    camera_theta += -(xpos - mouse_position_x)*0.005;
  }
  mouse_position_x = xpos;
  mouse_position_y = ypos;
}

void Render::mouse_pressed() {
  is_mouse_pressed = true;
}

void Render::mouse_released() {
  is_mouse_pressed = false;
}

void Render::mouse_scroll(double offset) {
  camera_radius += offset;
}

static void MessageCallback(GLenum source,
                     GLenum type,
                     GLuint id,
                     GLenum severity,
                     GLsizei length,
                     const GLchar* message,
                     const void* userParam)
{
    fprintf(stderr, "GL CALLBACK: %s type = 0x%x, severity = 0x%x, message = %s\n",
          (type == GL_DEBUG_TYPE_ERROR ? "** GL ERROR **" : ""),
          type, severity, message);
}

void error_callback(int error, const char* description) {
    throw std::runtime_error(description);
}

void Render::key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, 1);
}

void Render::scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    mouse_scroll(yoffset);
}

void Render::mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
        mouse_pressed();
    }
    if(button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE) {
        mouse_released();
    }
}

void Render::cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
    mouse_move(xpos, ypos);
}

void Render::setupWindow() {
    if (!glfwInit())
        throw std::runtime_error("Failed to initialize glfw");
    glfwSetErrorCallback(error_callback);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_SAMPLES, 4);

    this->window = glfwCreateWindow(height, width, "N-BODY SIMULATION", NULL, NULL);
    if (!this->window)
        throw std::runtime_error("Failed to create window");

    glfwMakeContextCurrent(this->window);
    glfwSetKeyCallback(this->window, this->key_callback);
    glfwSetCursorPosCallback(this->window, this->cursor_position_callback);
    glfwSetMouseButtonCallback(this->window, this->mouse_button_callback);
    glfwSetScrollCallback(this->window, this->scroll_callback);

    glfwSwapInterval(1); //enables v-sync
}

GLuint Render::load_shader(const char* path, int shader_type) {
    std::string shader_code;
    std::ifstream ShaderStream(path, std::ios::in);
    if(ShaderStream.is_open()){
        std::stringstream sstr;
        sstr << ShaderStream.rdbuf();
        shader_code = sstr.str();
        ShaderStream.close();
    }
    const char *shader_code_array[] = {shader_code.c_str()};
    GLuint shader = glCreateShader(shader_type);
    glShaderSource(shader, 1, shader_code_array, nullptr);
    glCompileShader(shader);
    GLint isCompiled = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &isCompiled);
    if(isCompiled == GL_FALSE)
    {
        GLint maxLength = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &maxLength);
        char* errorLog = new char[maxLength];
        glGetShaderInfoLog(shader, maxLength, &maxLength, errorLog);
        printf("Shader compilation failed:\n");
        printf("%s\n", errorLog);
        delete errorLog;
        glDeleteShader(shader);
        exit(1);
    }
    return shader;
  }

void Render::createAndBindBuffer() {
    glEnable(GL_DEBUG_OUTPUT);
    glDebugMessageCallback( (GLDEBUGPROC) MessageCallback, 0);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glEnable (GL_BLEND);
    glBlendFunc (GL_ONE, GL_ONE);
    //glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glGenBuffers(3, buffer);
    glBindBuffer(GL_ARRAY_BUFFER, buffer[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float)*3*numberOfBodies, positionsToRender, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

    glBindBuffer(GL_ARRAY_BUFFER, buffer[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(char)*3*numberOfBodies, colorsToRender, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_UNSIGNED_BYTE, GL_TRUE, 0, 0);
    //glPointSize(4.0);

    glBindBuffer(GL_ARRAY_BUFFER, buffer[2]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float)*numberOfBodies, massesToRender, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 0, 0);
}
// "    gl_PointSize = gl_Normal.z*((weight+100.0)/70.0);"

void Render::createAndCompileShaders() {
    shVertex = load_shader("./src/vertexshader.glsl", GL_VERTEX_SHADER);
    shFragment = load_shader("./src/fragmentshader.glsl", GL_FRAGMENT_SHADER);
}

void Render::createAndLinkProgram() {
    program = glCreateProgram();
    glAttachShader(program, shVertex);
    glAttachShader(program, shFragment);
    glBindAttribLocation(program, 0, "position");
    glBindAttribLocation(program, 1, "color");
    glBindAttribLocation(program, 2, "mass");
    glBindFragDataLocation (program, 0, "vertex_color");
    glLinkProgram(program);
    glDetachShader(program, shVertex);
    glDetachShader(program, shFragment);
}

void Render::init() {
    createAndBindBuffer();
    createAndCompileShaders();
    createAndLinkProgram();
}

void Render::setupOpenGL() {
    setupWindow();
    init();
}

bool Render::clearWindow() {
    if(!glfwWindowShouldClose(this->window)) {
        int wid, hei;
        glfwGetFramebufferSize(this->window, &wid, &hei);
        glViewport(0, 0, wid, hei);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        return false;
    }
    else {
        destroyWindow();
        return true;
    }
}

bool Render::swapBuffers() {
    if(!glfwWindowShouldClose(this->window)) {
        glfwSwapBuffers(this->window);
        glfwPollEvents();
        return false;
    }
    else {
        destroyWindow();
        return true;
    }
}

void Render::destroyWindow() {
    glfwDestroyWindow(this->window);
    glfwTerminate();
}

void Render::render() {
    glUseProgram(program);
    glm::mat4 view = glm::lookAt(
        glm::vec3(
        camera_radius*sin(camera_theta),
        camera_radius*cos(camera_theta)*cos(camera_phi),
        camera_radius*cos(camera_theta)*sin(camera_phi)
        ), // the position of your camera, in world space
        glm::vec3(0, 0, 0),   // where you want to look at, in world space
        glm::vec3(0, 1, 0)        // probably glm::vec3(0,1,0), but (0,-1,0) would make you looking upside-down, which can be great too
    );
    glm::mat4 projection = glm::perspective(
        glm::radians(45.0f), // The vertical Field of View, in radians: the amount of "zoom". Think "camera lens". Usually between 90° (extra wide) and 30° (quite zoomed in)
        1.0f*width/height,       // Aspect Ratio. Depends on the size of your window. Notice that 4/3 == 800/600 == 1280/960, sounds familiar ?
        0.1f,              // Near clipping plane. Keep as big as possible, or you'll get precision issues.
        100.0f             // Far clipping plane. Keep as little as possible.
    );
    glm::mat4 mvp = projection * view;
    GLuint MatrixID = glGetUniformLocation(program, "MVP");
    glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &mvp[0][0]);

    glBindBuffer(GL_ARRAY_BUFFER, buffer[0]);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float)*3*numberOfBodies, positionsToRender);

    glBindBuffer(GL_ARRAY_BUFFER, buffer[1]);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(char)*3*numberOfBodies, colorsToRender);

    glBindBuffer(GL_ARRAY_BUFFER, buffer[2]);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float)*numberOfBodies, massesToRender);

    glDrawArrays(GL_POINTS, 0, numberOfBodies);
}

bool Render::draw(thrust::host_vector<float>& positions) {
    for(unsigned i=0; i<numberOfBodies; i++) {
      for(int j=0; j<3; j++) {
        positionsToRender[i*3+j] = positions[i*3+j];
      }
    }
    bool closePressed = clearWindow();
    if(closePressed) return true;

    render();

    closePressed = swapBuffers();
    if(closePressed) return true;
    return false;
}

float Render::getTime() {
    return glfwGetTime();
}
