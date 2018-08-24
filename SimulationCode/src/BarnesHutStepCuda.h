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
    BarnesHutStepCuda(std::vector<double>& masses, unsigned numberOfBodies);
    ~BarnesHutStepCuda();
    void compute(tf3 &positions, double dt);

private:
    //NodeBH* root;
    ComputationsBarnesHut *c;
    std::vector<double> weights;
    std::vector<double> forces;
    std::vector<double> velocities;
    double sizeFrame = 8.0;
    const double theta = 1;
    const double EPS = 0.01;
};

#endif
