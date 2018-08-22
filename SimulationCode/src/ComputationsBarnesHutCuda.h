#ifndef CUDABARNESHUTSTEP_H
#define CUDABARNESHUTSTEP_H

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

typedef thrust::host_vector<float> type;

class ComputationsBarnesHut {
  thrust::device_vector<float> veloD;
  thrust::device_vector<float> weightsD;
  float* d_positions;
  float *d_velocities;
  float *d_weights;
  int numberOfBodies;

public:
  ComputationsBarnesHut(type velocities, type weights) {
    veloD = velocities;
    weightsD = weights;
    d_velocities = thrust::raw_pointer_cast(veloD.data());
    d_weights = thrust::raw_pointer_cast(weightsD.data());
  }
  ~ComputationsBarnesHut() {}
  void createTree(int numberOfBodies);
  void BarnesHutBridge(type &pos, int N, float dt);
};

#endif