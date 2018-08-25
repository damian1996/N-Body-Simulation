#ifndef CUDASTEP_H
#define CUDASTEP_H

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

typedef thrust::host_vector<float> type;

class Computations {
  thrust::device_vector<float> veloD;
  thrust::device_vector<float> weightsD;
  float *d_velocities;// = thrust::raw_pointer_cast(veloD.data());
  float *d_weights;// = thrust::raw_pointer_cast(weightsD.data());
  bool firstStep;
  bool oldMomentum[3];

public:
  Computations(type velocities, type weights) {
    veloD = velocities;
    weightsD = weights;
    d_weights = thrust::raw_pointer_cast(weightsD.data());
    d_velocities = thrust::raw_pointer_cast(veloD.data());
    firstStep = false;
  }
  ~Computations() {}
  void NaiveSimBridgeThrust(type &pos, int N, float dt);
  bool testingMomemntum(int N);
};

#endif
