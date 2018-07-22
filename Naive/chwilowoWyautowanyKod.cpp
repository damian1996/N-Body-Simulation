template <typename T> struct KernelArray {
  T *arr;
  int _size;
  KernelArray(int N) { arr = (T *)malloc(N * sizeof(T)); }
  KernelArray(thrust::device_vector<T> &dVec, int N) {
    arr = (T *)malloc(N * sizeof(T));
    arr = thrust::raw_pointer_cast(dVec.data());
    _size = (int)N;
  }
  ~KernelArray() { free(arr); }
};

// Function to convert device_vector to structure

template <typename T>
KernelArray<T> *cTK(thrust::device_vector<T> &dVec, int N) {
  KernelArray<T> *kArray = new KernelArray<T>(N);
  // kArray->arr = thrust::raw_pointer_cast(&dVec[0]);
  kArray->arr = thrust::raw_pointer_cast(dVec.data());
  kArray->_size = (int)N;

  return kArray;
}

const char *vertex_shader[] = {"#version 130\n"
                               "in vec3 position;\n"
                               "in vec3 color;\n"
                               "in float weight;\n"
                               "out vec3 vertex_color;\n"
                               "void main() {\n"
                               "    gl_Position = vec4(position,1.0);\n"
                               "    vertex_color=color;\n"
                               "}"};

const char *fragment_shader[] = {"#version 130\n"
                                 "in vec3 vertex_color;\n"
                                 "void main() {\n"
                                 "    gl_FragColor = vec4(vertex_color, 1);\n"
                                 "}"};

glEnable(GL_DEBUG_OUTPUT);
glDebugMessageCallback((GLDEBUGPROC)MessageCallback, 0);
glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
glGenBuffers(3, buffer);

...

    glBindBuffer(GL_ARRAY_BUFFER, buffer[2]);
glBufferData(GL_ARRAY_BUFFER, sizeof(float) * N, V_weight, GL_DYNAMIC_DRAW);
glEnableVertexAttribArray(2);
glVertexAttribPointer(2, 3, GL_FLOAT, GL_TRUE, 0, 0);

...

    glBindAttribLocation(program, 2, "weight");

...

    glBindBuffer(GL_ARRAY_BUFFER, buffer[2]);
glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float) * N, V_weight);

/*
void NaiveSimBridge(Render* painter, type& pos, type& velocities, type& weights,
int N) {
    // type = thrust::device_vector<float>

    thrust::device_vector<float> posD = pos;
    thrust::device_vector<float> veloD = velocities;
    thrust::device_vector<float> weightsD = weights;

    while(true) {
        KernelArray<float>* d_positions = cTK<float>(posD, 3*N);
        KernelArray<float>* d_velocities = cTK<float>(veloD, 3*N);
        KernelArray<float>* d_weights = cTK<float>(weightsD, N);

        //NaiveSim<<<(N+255)/256, 256>>>(cTK<float>(d_positions, 3*N),
cTK<float>(d_velocities, 3*N), cTK<float>(d_weights, N));
        NaiveSim<<<8, (N+7)/8>>>(d_positions, d_velocities, d_weights);
        cudaDeviceSynchronize();
        // potrzebuje wskazniczkow z powrotem

        // wrap raw pointer with a device_ptr
        thrust::device_ptr<float> dev_ptr1(d_positions->arr);
        thrust::device_ptr<float> dev_ptr2(d_velocities->arr);
        thrust::device_ptr<float> dev_ptr3(d_weights->arr);

        // copy memory to a new device_vector (which automatically allocates
memory)
        thrust::device_vector<float> vec1(dev_ptr1, dev_ptr1 + 3*N);
        thrust::device_vector<float> vec2(dev_ptr2, dev_ptr2 + 3*N);
        thrust::device_vector<float> vec3(dev_ptr3, dev_ptr3 + N);

        break;
        //painter->sendData(positions);
        //if(painter->draw()) break;
    }

    // free user-allocated memory
    //cudaFree(raw_ptr);

    cudaMallocManaged(&tx, sizeof(int*));
    cudaMallocManaged(&ty, sizeof(int*));
    *tx = x;
    *ty = x;
    NaiveSim<<<,1>>>(tx, ty);
    cudaDeviceSynchronize();
    x = *tx;
}
*/
