#ifndef STEPNAIVECUDA_H
#define STEPNAIVECUDA_H

#include <vector>

#include "RandomGenerators.h"

#include "Step.h"
#include "ComputationsCuda.h"

class StepNaiveCuda : public Step {
  Computations* c;
  RandomGenerators* rg;
  thrust::host_vector<double> weights;
  thrust::host_vector<double> velocities;
public:
    StepNaiveCuda(std::vector<float> masses, unsigned N);
    ~StepNaiveCuda();
    void initialize() {

    }
    void compute(tf3& positions, double dt);
};

#endif
