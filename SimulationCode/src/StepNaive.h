#ifndef STEPNAIVE_H
#define STEPNAIVE_H

#include "Step.h"
#include <cassert>
#include <cmath>
#include <iterator>

class StepNaive : public Step {
  std::vector<double> forces;
  std::vector<double> velocities;
  std::vector<double> weights;
  bool firstStep;
public:
  StepNaive(std::vector<double> &masses, unsigned numberOfBodies);
  ~StepNaive();
  bool testingMomemntum();
  void compute(tf3 &positions, double dt);
};

#endif
