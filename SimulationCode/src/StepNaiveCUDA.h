#ifndef STEPNAIVECUDA_H
#define STEPNAIVECUDA_H

#include <vector>

#include "RandomGenerators.h"

#include "ComputationsCuda.h"
#include "Step.h"

class StepNaiveCuda : public Step {
  Computations *c;
  RandomGenerators *randomGenerator;
  thrust::host_vector<double> weights;
  thrust::host_vector<double> velocities;

public:
  StepNaiveCuda(std::vector<double> masses, unsigned numberOfBodies);
  ~StepNaiveCuda();
  void compute(tf3 &positions, double dt);
};

#endif
