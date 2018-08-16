#ifndef CUDABARNESHUTSTEP_H
#define CUDABARNESHUTSTEP_H

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

typedef thrust::host_vector<float> type;

class ComputationsBarnesHut {
  thrust::device_vector<float> veloD;
  thrust::device_vector<float> weightsD;

public:
  ComputationsBarnesHut(type velocities, type weights) {
    veloD = velocities;
    weightsD = weights;
  }
  ~ComputationsBarnesHut() {}
  void BarnesHutBridge(type &pos, int N, float dt);
};

#endif
