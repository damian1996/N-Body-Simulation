#ifndef CUDASTEP_H
#define CUDASTEP_H

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

typedef thrust::host_vector<float> type;

class Computations {
  thrust::device_vector<float> veloD;
  thrust::device_vector<float> weightsD;

public:
  Computations(type velocities, type weights) {
    veloD = velocities;
    weightsD = weights;
  }
  ~Computations() {}
  // void NaiveSimBridge(Render* painter, type& pos, type& velocities, type&
  // weights, int N);
  void NaiveSimBridgeThrust(type &pos, int N, float dt);
};

#endif
