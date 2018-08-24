#ifndef CUDABARNESHUTSTEP_H
#define CUDABARNESHUTSTEP_H

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <algorithm>

typedef thrust::host_vector<double> type;

class ComputationsBarnesHut {
  thrust::device_vector<double> veloD;
  thrust::device_vector<double> weightsD;
  double* d_positions;
  double *d_velocities;
  double *d_weights;
  int numberOfBodies;

public:
  ComputationsBarnesHut(type velocities, type weights) {
    veloD = velocities;
    weightsD = weights;
    d_velocities = thrust::raw_pointer_cast(veloD.data());
    d_weights = thrust::raw_pointer_cast(weightsD.data());
  }
  ~ComputationsBarnesHut() {}
  void createTree(int numberOfBodies, double dt);
  void BarnesHutBridge(type &pos, int N, double dt);
  bool testingMomemntum(int N);
};

#endif