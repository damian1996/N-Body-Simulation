#include "ComputationsBarnesHutCuda.h"

const float G = 6.674 * (1e-11);
const float EPS = 0.01f;

template <typename T>
__global__ void BarnesHutSim(T *pos, T *velo, T *weigh, int numberOfBodies, float dt) {
  int thid = blockIdx.x * blockDim.x + threadIdx.x;
  if(thid>numberOfBodies) return;
}

void ComputationsBarnesHut::BarnesHutBridge(type &pos, int numberOfBodies, float dt) {
  thrust::device_vector<float> posD = pos;
  float *d_positions = thrust::raw_pointer_cast(posD.data());
  float *d_velocities = thrust::raw_pointer_cast(veloD.data());
  float *d_weights = thrust::raw_pointer_cast(weightsD.data());
  BarnesHutSim<<<64, (numberOfBodies + 63) / 64>>>(d_positions, d_velocities, d_weights, numberOfBodies, dt);
  pos = posD;
}
