#ifndef BARNESHUTSTEPCUDA_H
#define BARNESHUTSTEPCUDA_H

#include "Step.h"
#include "ComputationsBarnesHutCuda.h"
#include "RandomGenerators.h"
#include <thrust/host_vector.h>
#include <algorithm>
#include <array>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

class BarnesHutStepCuda : public Step {
public:
    BarnesHutStepCuda(std::vector<float>& masses, unsigned numberOfBodies);
    ~BarnesHutStepCuda();
    void compute(tf3 &positions, float dt);

private:
    //NodeBH* root;
    ComputationsBarnesHut *c;
    std::vector<float> weights;
    std::vector<float> forces;
    std::vector<float> velocities;
    float sizeFrame = 8.0;
    const float theta = 1;
    const float EPS = 0.01;
};

#endif
