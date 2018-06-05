#include "Simulation.cuh"

const float G = 6.674*(1e-10);
const float dt = 0.016f;

template <typename T>
__global__
void NaiveSim(T* pos, T* velo, T* weigh, int N) {
    //printf("%f\n", pos->arr[thid]);
    int thid = blockIdx.x*blockDim.x + threadIdx.x;
    float posx = pos[thid*3], posy = pos[thid*3+1], weighI = weigh[thid];
    float forcex = 0.0f, forcey = 0.0f;
    for(int j=0; j<N; j++) {
        if(j!=thid) {
            float distX = pos[j*3] - posx;
            float distY = pos[j*3+1] - posy;
            if(fabs(distX)>1e-10 && fabs(distY)>1e-10) {
                float F = G*(weighI*weigh[j]);
                forcex += F*distX/(distX*distX+distY*distY);
                forcey += F*distX/(distX*distX+distY*distY);
            }
        }
    }
    float acc = forcex/weighI;
    pos[thid*3] += velo[thid*3]*dt + acc*dt*dt/2;
    velo[thid*3] += acc*dt;

    acc = forcey/weighI;
    pos[thid*3+1] += velo[thid*3+1]*dt + acc*dt*dt/2;
    velo[thid*3+1] += acc*dt;
    __syncthreads();

    /*
        printf("%d\n", thid);
        //float force = 0.0f;
        // liczymy force
        for(int i=0; i<weigh->_size; i++) {
          //weigh->arr[thid] += 100.0f;
        }
        // wyliczamy nowa pozycje i accelerate
        // synchronize
        // wyliczamy nowe velocities

    */
}

void NaiveSimBridgeThrust(Scene* painter, type& pos, type& velocities, type& weights, int N) {
    // type = thrust::device_vector<float>
    thrust::device_vector<float> posD = pos;
    thrust::device_vector<float> veloD = velocities;
    thrust::device_vector<float> weightsD = weights;

    //thrust::copy(weightsD.begin(), weightsD.end(), std::ostream_iterator<float>(std::cout, " "));
    std::cout << std::endl;
    while(true) {
        float* d_positions = thrust::raw_pointer_cast(posD.data());
        float* d_velocities = thrust::raw_pointer_cast(veloD.data());
        float* d_weights = thrust::raw_pointer_cast(weightsD.data());

        //NaiveSim<<<(N+255)/256, 256>>>(cTK<float>(d_positions, 3*N), cTK<float>(d_velocities, 3*N), cTK<float>(d_weights, N));
        NaiveSim<<<64, (N+63)/64>>>(d_positions, d_velocities, d_weights, N);

        //thrust::copy(weightsD.begin(), weightsD.end(), std::ostream_iterator<float>(std::cout, " "));

        pos = posD;
        painter->sendData(pos);
        if(painter->draw()) break;
    }

    // free user-allocated memory
    //cudaFree(raw_ptr);

    /*
    cudaMallocManaged(&tx, sizeof(int*));
    cudaMallocManaged(&ty, sizeof(int*));
    *tx = x;
    *ty = x;
    NaiveSim<<<,1>>>(tx, ty);
    cudaDeviceSynchronize();
    x = *tx;
    */
}

/*
void NaiveSimBridge(Scene* painter, type& pos, type& velocities, type& weights, int N) {
    // type = thrust::device_vector<float>

    thrust::device_vector<float> posD = pos;
    thrust::device_vector<float> veloD = velocities;
    thrust::device_vector<float> weightsD = weights;

    while(true) {
        KernelArray<float>* d_positions = cTK<float>(posD, 3*N);
        KernelArray<float>* d_velocities = cTK<float>(veloD, 3*N);
        KernelArray<float>* d_weights = cTK<float>(weightsD, N);

        //NaiveSim<<<(N+255)/256, 256>>>(cTK<float>(d_positions, 3*N), cTK<float>(d_velocities, 3*N), cTK<float>(d_weights, N));
        NaiveSim<<<8, (N+7)/8>>>(d_positions, d_velocities, d_weights);
        cudaDeviceSynchronize();
        // potrzebuje wskazniczkow z powrotem

        // wrap raw pointer with a device_ptr
        thrust::device_ptr<float> dev_ptr1(d_positions->arr);
        thrust::device_ptr<float> dev_ptr2(d_velocities->arr);
        thrust::device_ptr<float> dev_ptr3(d_weights->arr);

        // copy memory to a new device_vector (which automatically allocates memory)
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

// https://www.bu.edu/pasi/files/2011/07/Lecture6.pdf
// https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
// https://stackoverflow.com/questions/4176762/passing-structs-to-cuda-kernels
// https://codeyarns.com/2011/02/16/cuda-dim3/
// http://developer.download.nvidia.com/CUDA/training/introductiontothrust.pdf
// https://groups.google.com/forum/#!topic/thrust-users/4EaWLGeJOO8
// https://github.com/thrust/thrust/blob/master/examples/cuda/unwrap_pointer.cu
