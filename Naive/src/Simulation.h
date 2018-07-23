#ifndef SIMULATION_H
#define SIMULATION_H

#include <array>
#include <cstdio>
#include <stdexcept>
#include <unistd.h>
#include <vector>

#include "RandomGenerators.h"
#include "Render.h"
#include "Step.h"

class Simulation {
  Render *r;
  Step *comp;
  unsigned N;
  thrust::host_vector<float> positions;
  RandomGenerators *rg;

public:
  Simulation(Render *r, Step *comp, unsigned N);
  ~Simulation();
  void makeSimulation();
};

#endif
