#include "ComputationsCuda.h"

const float G = 6.674 * (1e-11);
const float EPS = 0.01f;

template <typename T>
__global__ void NaiveSim(T *pos, T *velo, T *weigh, int N, float dt) {
  // printf("%f\n", pos->arr[thid]);
  int thid = blockIdx.x * blockDim.x + threadIdx.x;
  float posx = pos[thid * 3], posy = pos[thid * 3 + 1],
        posz = pos[thid * 3 + 2], weighI = weigh[thid];
  float forcex = 0.0f, forcey = 0.0f, forcez = 0.0f;
  for (int j = 0; j < N; j++) {
    if (j != thid) {
      float distX = pos[j * 3] - posx;
      float distY = pos[j * 3 + 1] - posy;
      float distZ = pos[j * 3 + 2] - posz;
      float dist = (distX * distX + distY * distY + distZ * distZ) + EPS * EPS;
      if (fabs(distX) > 1e-10 && fabs(distY) > 1e-10 && fabs(distZ) > 1e-10) {
        float F = G * (weighI * weigh[j]);
        forcex += F * distX / dist;
        forcey += F * distY / dist;
        forcez += F * distZ / dist;

        /*if(dist>=EPS) {
            forcex += F*distX/dist;
            forcey += F*distY/dist;
        } else {
          forcex += F*distX;
          forcey += F*distY;
        }*/
      }
    }
  }
  float acc = forcex / weighI;
  pos[thid * 3] += velo[thid * 3] * dt + acc * dt * dt / 2;
  velo[thid * 3] += acc * dt;

  acc = forcey / weighI;
  pos[thid * 3 + 1] += velo[thid * 3 + 1] * dt + acc * dt * dt / 2;
  velo[thid * 3 + 1] += acc * dt;

  acc = forcez / weighI;
  pos[thid * 3 + 2] += velo[thid * 3 + 2] * dt + acc * dt * dt / 2;
  velo[thid * 3 + 2] += acc * dt;

  __syncthreads();
}

void Computations::NaiveSimBridgeThrust(type &pos, int N, float dt) {
  // type = thrust::device_vector<float>
  thrust::device_vector<float> posD = pos;

  float *d_positions = thrust::raw_pointer_cast(posD.data());
  float *d_velocities = thrust::raw_pointer_cast(veloD.data());
  float *d_weights = thrust::raw_pointer_cast(weightsD.data());

  float zzpx = 0.0f, zzpy = 0.0f, zzpz = 0.0f;
  for (int i = 0; i < N; i++) {
    zzpx += (weightsD[i] * veloD[i * 3]);
    zzpy += (weightsD[i] * veloD[i * 3 + 1]);
    zzpz += (weightsD[i] * veloD[i * 3 + 2]);
  }
  // std::cout << "Pedy : " << zzpx << "  " << zzpy << std::endl;

  NaiveSim<<<64, (N + 63) / 64>>>(d_positions, d_velocities, d_weights, N, dt);

  pos = posD;
  // thrust::copy(weightsD.begin(), weightsD.end(),
  // std::ostream_iterator<float>(std::cout, " "));
}

// https://www.bu.edu/pasi/files/2011/07/Lecture6.pdf
// https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
// https://stackoverflow.com/questions/4176762/passing-structs-to-cuda-kernels
// https://codeyarns.com/2011/02/16/cuda-dim3/
// http://developer.download.nvidia.com/CUDA/training/introductiontothrust.pdf
// https://groups.google.com/forum/#!topic/thrust-users/4EaWLGeJOO8
// https://github.com/thrust/thrust/blob/master/examples/cuda/unwrap_pointer.cu
