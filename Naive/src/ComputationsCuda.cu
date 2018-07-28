#include "ComputationsCuda.h"

const float G = 6.674 * (1e-11);
const float EPS = 0.01f;

template <typename T>
__global__ void NaiveSim(T *pos, T *velo, T *weigh, int numberOfBodies, float dt) {
  int thid = blockIdx.x * blockDim.x + threadIdx.x;
  float posX = pos[thid * 3], posY = pos[thid * 3 + 1], posZ = pos[thid * 3 + 2];
  float weighI = weigh[thid];
  float force[3] = {0.0f, 0.0f, 0.0f};

  for (int j = 0; j < numberOfBodies; j++) {
    if (j != thid) {
      float distX = pos[j * 3] - posX;
      float distY = pos[j * 3 + 1] - posY;
      float distZ = pos[j * 3 + 2] - posZ;

      float dist = (distX * distX + distY * distY + distZ * distZ) + EPS * EPS;
      dist = dist*sqrt(dist);
      float F = G * (weighI * weigh[j]);
      force[0] += F*distX/dist;
      force[1] += F*distY/dist;
      force[2] += F*distZ/dist;
    }
  }
  for(int k=0; k<3; k++) {
    float acc = force[k] / weighI;
    pos[thid*3+k] += velo[thid*3+k]*dt + acc*dt*dt/2;
    velo[thid*3+k] += acc*dt;
  }
}

bool Computations::testingMomemntum(int numberOfBodies) {
  float momentum[3] = {0.0f, 0.0f, 0.0f};
  for (unsigned i = 0; i < numberOfBodies; i++) {
      for(int k = 0; k < 3; k++) {
          momentum[k] += (weightsD[i] * veloD[i*3 + k]);
      }
  }
  if(!firstStep) {
    firstStep = true;
    for(int k=0; k<3; k++) oldMomentum[k] = momentum[k];
  }
  for(int k=0; k<3; k++) {
    if(fabs(oldMomentum[k] - momentum[k]) > 1.0) {
      for(int k=0; k<3; k++) std::cout << momentum[k] << " \n"[k == 2];
      return false;
    }
    oldMomentum[k] = momentum[k];
  }
  return true;
}

void Computations::NaiveSimBridgeThrust(type &pos, int numberOfBodies, float dt) {
  thrust::device_vector<float> posD = pos;
  float *d_positions = thrust::raw_pointer_cast(posD.data());
  float *d_velocities = thrust::raw_pointer_cast(veloD.data());
  float *d_weights = thrust::raw_pointer_cast(weightsD.data());
  NaiveSim<<<64, (numberOfBodies + 63) / 64>>>(d_positions, d_velocities, d_weights, numberOfBodies, dt);
  pos = posD;
}

// https://www.bu.edu/pasi/files/2011/07/Lecture6.pdf
// https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
// https://stackoverflow.com/questions/4176762/passing-structs-to-cuda-kernels
// https://codeyarns.com/2011/02/16/cuda-dim3/
// http://developer.download.nvidia.com/CUDA/training/introductiontothrust.pdf
// https://groups.google.com/forum/#!topic/thrust-users/4EaWLGeJOO8
// https://github.com/thrust/thrust/blob/master/examples/cuda/unwrap_pointer.cu
// thrust::copy(weightsD.begin(), weightsD.end(),
// std::ostream_iterator<float>(std::cout, " "));
//testingMomemntum(numberOfBodies);
//if(!testingMomemntum(numberOfBodies)) std::cout << "problemiki" << "\n";
