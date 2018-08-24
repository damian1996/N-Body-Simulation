#ifndef STEP_H
#define STEP_H

#include "RandomGenerators.h"
#include <thrust/host_vector.h>

typedef thrust::host_vector<double> tf3;

class Step {
protected:
  const double G = 6.674 * (1e-11);
  double oldMomentumX = 0, oldMomentumY = 0, oldMomentumZ = 0;
  RandomGenerators *randomGenerator;
  unsigned numberOfBodies;

public:
  virtual ~Step() {}
  virtual void compute(tf3 &positions, double dt) = 0;
};

#endif
