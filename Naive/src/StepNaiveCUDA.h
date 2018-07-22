#ifndef STEPNAIVECUDA_H
#define STEPNAIVECUDA_H

#include <vector>

#include "RandomGenerators.h"

#include "Step.h"
#include "ComputationsCuda.h"

class StepNaiveCuda : public Step {
  Computations* c;
  RandomGenerators* rg;
  thrust::host_vector<float> weights;
  thrust::host_vector<float> velocities;
public:
    StepNaiveCuda(std::vector<float> masses, unsigned N);
    ~StepNaiveCuda();
    void initialize() {

    }
    void compute(tf3& positions, float dt);
};

#endif
