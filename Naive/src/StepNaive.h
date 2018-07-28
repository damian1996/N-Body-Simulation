#ifndef STEPNAIVE_H
#define STEPNAIVE_H

#include "Step.h"
#include <cassert>
#include <cmath>
#include <iterator>

class StepNaive : public Step {
  std::vector<float> forces;
  std::vector<float> velocities;
  std::vector<float> weights;
  bool firstStep;
public:
  StepNaive(std::vector<float> &masses, unsigned numberOfBodies);
  ~StepNaive();
  bool testingMomemntum();
  void compute(tf3 &positions, float dt);
};

#endif
