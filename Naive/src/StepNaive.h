#ifndef STEPNAIVE_H
#define STEPNAIVE_H

#include <cmath>
#include <iterator>
#include "Step.h"

class StepNaive  : public Step {
    std::vector<double> forces;
    std::vector<double> velocities;
    std::vector<double> weights;
public:
    StepNaive(std::vector<float>& masses, unsigned N);
    ~StepNaive();
    void initialize();
    void compute(tf3& positions, double dt);
};

#endif
