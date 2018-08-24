#include "ComputationsCuda.h"

const float G = 6.674 * (1e-11);
const float EPS = 0.01f;

template <typename T>
__global__ void NaiveSim(T *pos, T *velo, T *weigh, int numberOfBodies, float dt) {
  int thid = blockIdx.x * blockDim.x + threadIdx.x;
  if(thid>numberOfBodies) return;
  float posX = pos[thid * 3], posY = pos[thid * 3 + 1], posZ = pos[thid * 3 + 2];
  float weighI = weigh[thid];
  float force[3] = {0.0f, 0.0f, 0.0f};
  for (int j = 0; j < thid; j++) {
      float distX = pos[j * 3] - posX;
      float distY = pos[j * 3 + 1] - posY;
      float distZ = pos[j * 3 + 2] - posZ;

      float dist = (distX * distX + distY * distY + distZ * distZ) + EPS * EPS;
      dist = dist*sqrt(dist);
      float F = G * (weighI * weigh[j]);
      float cosik = F/dist;
      force[0] += cosik*distX;
      force[1] += cosik*distY;
      force[2] += cosik*distZ;
  }
  for (int j = thid+1; j < numberOfBodies; j++) {
      float distX = pos[j * 3] - posX;
      float distY = pos[j * 3 + 1] - posY;
      float distZ = pos[j * 3 + 2] - posZ;

      float dist = (distX * distX + distY * distY + distZ * distZ) + EPS * EPS;
      dist = dist*sqrt(dist);
      float F = G * (weighI * weigh[j]);
      float cosik = F/dist;
      force[0] += cosik*distX;
      force[1] += cosik*distY;
      force[2] += cosik*distZ;
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
  std::cout << momentum[0] << " " << momentum[1] << " " << momentum[2] << std::endl;
  if(!firstStep) {
    firstStep = true;
    for(int k=0; k<3; k++) oldMomentum[k] = momentum[k];
  }
  for(int k=0; k<3; k++)
    if(fabs(oldMomentum[k] - momentum[k]) > 1.0)
       return false;

  for(int k=0; k<3; k++) oldMomentum[k] = momentum[k];
  for(int k=0; k<3; k++) std::cout << momentum[k] << " \n"[k == 2];
  return true;
}

void Computations::NaiveSimBridgeThrust(type &pos, int numberOfBodies, float dt) {
  thrust::device_vector<float> posD = pos;
  float *d_positions = thrust::raw_pointer_cast(posD.data());
  NaiveSim<<<(numberOfBodies+1023)/1024, 1024>>>(d_positions, d_velocities, d_weights, numberOfBodies, dt);
  testingMomemntum(numberOfBodies);
  pos = posD;
}
