#ifndef STEPNAIVE_H
#define STEPNAIVE_H

#include "Step.h"
#include <cmath>
#include <iterator>

class StepNaive : public Step {
  std::vector<float> forces;
  std::vector<float> velocities;
  std::vector<float> weights;

public:
  StepNaive(std::vector<float> &masses, unsigned N);
  ~StepNaive();
  void initialize();
  void compute(tf3 &positions, float dt);
};

#endif
