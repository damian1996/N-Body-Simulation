#ifndef STEPNAIVECUDA_H
#define STEPNAIVECUDA_H

#include "RandomGenerators.h"

#include "Step.h"
#include "ComputationsCuda.h"

class StepNaiveCuda : public Step {
  Computations* c;
  RandomGenerators* rg;
  thrust::host_vector<float> weights;
  thrust::host_vector<float> velocities;
public:
    StepNaiveCuda(unsigned N);
    ~StepNaiveCuda();
    void initialize() {

    }
    void compute(tf3& positions, float dt);
};

#endif
