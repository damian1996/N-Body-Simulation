#ifndef STEP_H
#define STEP_H

#include "RandomGenerators.h"
#include <thrust/host_vector.h>

typedef thrust::host_vector<float> tf3;

class Step {
protected:
  const float G = 6.674 * (1e-11);
  float oldMomentumX = 0, oldMomentumY = 0, oldMomentumZ = 0;
  RandomGenerators *randomGenerator;
  unsigned numberOfBodies;

public:
  virtual ~Step() {}
  virtual void compute(tf3 &positions, float dt) = 0;
};

#endif
