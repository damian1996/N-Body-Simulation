#ifndef STEP_H
#define STEP_H

#include "RandomGenerators.h"
#include <thrust/host_vector.h>

typedef thrust::host_vector<double> tf3;

class Step {
protected:
  const double G = 6.674*(1e-11);
  //const double x = (3.086 * 1e13);           //[km -> pc]
  //const double G = (4.3 * (1e-3)) / (x * x); //[pc^3/Mo*s^2]
  RandomGenerators *rg;
  unsigned N;

public:
  virtual ~Step() {}
  virtual void initialize() = 0;
  virtual void compute(tf3 &positions, double dt) = 0;
};

#endif
