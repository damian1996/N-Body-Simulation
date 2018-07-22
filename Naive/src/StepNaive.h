#ifndef STEPNAIVE_H
#define STEPNAIVE_H

#include "Step.h"
#include <algorithm>
#include <cmath>
#include <iterator>

class StepNaive : public Step {
  std::vector<double> forces;
  std::vector<double> velocities;
  std::vector<double> weights;

public:
  StepNaive(std::vector<double> &masses, unsigned N);
  ~StepNaive();
  void initialize();
  void compute(tf3 &positions, double dt);
};

#endif
