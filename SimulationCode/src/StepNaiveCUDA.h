#ifndef STEPNAIVECUDA_H
#define STEPNAIVECUDA_H

#include <vector>

#include "RandomGenerators.h"

#include "ComputationsCuda.h"
#include "Step.h"

class StepNaiveCuda : public Step {
  Computations *c;
  RandomGenerators *randomGenerator;
  thrust::host_vector<float> weights;
  thrust::host_vector<float> velocities;

public:
  StepNaiveCuda(std::vector<float> masses, unsigned numberOfBodies);
  ~StepNaiveCuda();
  void compute(tf3 &positions, float dt);
};

#endif
