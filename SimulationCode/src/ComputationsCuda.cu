#include "ComputationsCuda.h"

const double G = 6.674 * (1e-11);
const double EPS = 0.01f;

template <typename T>
__global__ void NaiveSim(T *pos, T *velo, T *weigh, int numberOfBodies, double dt) {
  int thid = blockIdx.x * blockDim.x + threadIdx.x;
  if(thid>numberOfBodies) return;
  double posX = pos[thid * 3], posY = pos[thid * 3 + 1], posZ = pos[thid * 3 + 2];
  double weighI = weigh[thid];
  double force[3] = {0.0f, 0.0f, 0.0f};
  for (int j = 0; j < thid; j++) {
      double distX = pos[j * 3] - posX;
      double distY = pos[j * 3 + 1] - posY;
      double distZ = pos[j * 3 + 2] - posZ;

      double dist = (distX * distX + distY * distY + distZ * distZ) + EPS * EPS;
      dist = dist*sqrt(dist);
      double F = G * (weighI * weigh[j]);
      double cosik = F/dist;
      force[0] += cosik*distX;
      force[1] += cosik*distY;
      force[2] += cosik*distZ;
  }
  for (int j = thid+1; j < numberOfBodies; j++) {
      double distX = pos[j * 3] - posX;
      double distY = pos[j * 3 + 1] - posY;
      double distZ = pos[j * 3 + 2] - posZ;

      double dist = (distX * distX + distY * distY + distZ * distZ) + EPS * EPS;
      dist = dist*sqrt(dist);
      double F = G * (weighI * weigh[j]);
      double cosik = F/dist;
      force[0] += cosik*distX;
      force[1] += cosik*distY;
      force[2] += cosik*distZ;
  }
  for(int k=0; k<3; k++) {
    double acc = force[k] / weighI;
    pos[thid*3+k] += velo[thid*3+k]*dt + acc*dt*dt/2;
    velo[thid*3+k] += acc*dt;
  }
}

bool Computations::testingMomemntum(int numberOfBodies) {
  double momentum[3] = {0.0f, 0.0f, 0.0f};
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

void Computations::NaiveSimBridgeThrust(type &pos, int numberOfBodies, double dt) {
  thrust::device_vector<double> posD = pos;
  double *d_positions = thrust::raw_pointer_cast(posD.data());
  NaiveSim<<<(numberOfBodies+1023)/1024, 1024>>>(d_positions, d_velocities, d_weights, numberOfBodies, dt);
  testingMomemntum(numberOfBodies);
  pos = posD;
}
