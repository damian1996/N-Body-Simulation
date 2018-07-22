#ifndef STEP_H
#define STEP_H

#include "RandomGenerators.h"
#include <thrust/host_vector.h>

typedef thrust::host_vector<float > tf3;

class Step {
protected:
    const float G = 6.674*(1e-11);
    //const float dt = 0.016f;
    RandomGenerators* rg;
    unsigned N;
public:
    virtual ~Step() {}
    virtual void initialize() = 0;
    virtual void compute(tf3& positions, float dt) = 0;
};

#endif
