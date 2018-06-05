#include "render.h"
#include <iostream>
Scene::Scene(unsigned N) : N(N) {
  opengl = new initOpenGL();
  V_position = new float3d[N];
  V_color = new byte3d[N];
  V_weight = new float[N];
}

Scene::~Scene() {
  delete [] V_weight;
  delete [] V_color;
  delete [] V_position;
  delete opengl;
}

void MessageCallback( GLenum source, GLenum type, GLuint id, GLenum severity,
     GLsizei length, const GLchar* message, const void* userParam)
{
     fprintf( stderr, "GL CALLBACK: %s type = 0x%x, severity = 0x%x, message = %s\n",
        ( type == GL_DEBUG_TYPE_ERROR ? "** GL ERROR **" : "" ),
        type, severity, message );
}

void Scene::createAndBindBuffer() {
    glEnable(GL_DEBUG_OUTPUT);
    glDebugMessageCallback( (GLDEBUGPROC) MessageCallback, 0 );
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
    glGenBuffers(3, buffer);
    glBindBuffer(GL_ARRAY_BUFFER, buffer[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float)*3*N, V_position, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glBindBuffer(GL_ARRAY_BUFFER, buffer[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(char)*3*N, V_color, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_UNSIGNED_BYTE, GL_TRUE, 0, 0);

    glBindBuffer(GL_ARRAY_BUFFER, buffer[2]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float)*N, V_weight, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_TRUE, 0, 0);

    glPointSize(2.0);
}
// "    gl_PointSize = gl_Normal.z*((weight+100.0)/70.0);"

void Scene::createAndCompileShaders() {
    const char* vertex_shader[] = {
        "#version 130\n"
        "in vec3 position;\n"
        "in vec3 color;\n"
        "in float weight;\n"
        "out vec3 vertex_color;\n"
        "void main() {\n"
        "    gl_Position = vec4(position,1.0);\n"
        "    vertex_color=color;\n"
        "}"
    };

    const char* fragment_shader[] = {
        "#version 130\n"
        "in vec3 vertex_color;\n"
        "void main() {\n"
        "    gl_FragColor = vec4(vertex_color, 1);\n"
        "}"
    };

    sh_vertex = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(sh_vertex, 1, vertex_shader, nullptr);
    glCompileShader(sh_vertex);
    GLint isCompiled = 0;
    glGetShaderiv(sh_vertex, GL_COMPILE_STATUS, &isCompiled);
    if(isCompiled == GL_FALSE)
    {
    	GLint maxLength = 0;
    	glGetShaderiv(sh_vertex, GL_INFO_LOG_LENGTH, &maxLength);
        char* errorLog = new char[maxLength];
    	glGetShaderInfoLog(sh_vertex, maxLength, &maxLength, errorLog);
        printf("Vertex shader compilation failed\n");
        printf("%s\n", errorLog);
        delete errorLog;
    	glDeleteShader(sh_vertex); // Don't leak the shader.
    	exit(1);
    }

    sh_fragment = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(sh_fragment, 1, fragment_shader, nullptr);
    glCompileShader(sh_fragment);
}

void Scene::createAndLinkProgram() {
    program = glCreateProgram();
    glAttachShader(program, sh_vertex);
    glAttachShader(program, sh_fragment);
    glBindAttribLocation(program, 0, "position");
    glBindAttribLocation(program, 1, "color");
    glBindAttribLocation(program, 2, "weight");
    glBindFragDataLocation (program, 0, "vertex_color");
    glLinkProgram(program);
    glDetachShader(program, sh_vertex);
    glDetachShader(program, sh_fragment);
}

void Scene::init() {
    createAndBindBuffer();
    createAndCompileShaders();
    createAndLinkProgram();
}

void Scene::render() {
    glUseProgram(program);
    glBindBuffer(GL_ARRAY_BUFFER, buffer[0]);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float)*3*N, V_position);
    glBindBuffer(GL_ARRAY_BUFFER, buffer[1]);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(char)*3*N, V_color);
    glBindBuffer(GL_ARRAY_BUFFER, buffer[2]);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float)*N, V_weight);
    glDrawArrays(GL_POINTS, 0, N);
}

void Scene::setupOpenGL() {
    opengl->init();
    opengl->setupWindow();
    init();
}

bool Scene::draw() {
    bool res = opengl->ClearWindow();
    if(res) return true;
    render();
    res = opengl->Swap();
    if(res) return true;
    return false;
}

void Scene::sendInitialData(tf3 positions, b3 colors, tf3 weights) {
    for(unsigned i=0; i<N; i++) {
        for(unsigned j=0; j<3; j++) {
            V_position[i][j] = positions[i*3+j]; // [i, j] = i*3 + j
            V_color[i][j] = colors[i][j];
        }
        V_weight[i] = weights[i];
    }
}

void Scene::sendData(tf3 positions) {
    for(unsigned i=0; i<N; i++) {
        for(unsigned j=0; j<3; j++)
            V_position[i][j] = positions[i*3+j]; // [i, j] = i*3 + j
    }
}
