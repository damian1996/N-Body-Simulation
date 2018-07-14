#ifndef CUDASTEP_H
#define CUDASTEP_H

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

typedef thrust::host_vector<double> type;

class Computations {
  thrust::device_vector<double> veloD;
  thrust::device_vector<double> weightsD;
public:
  Computations(type velocities, type weights) {
      veloD = velocities;
      weightsD = weights;
  }
  ~Computations() {

  }
  //void NaiveSimBridge(Render* painter, type& pos, type& velocities, type& weights, int N);
  void NaiveSimBridgeThrust(type& pos, int N, double dt);
};

#endif
