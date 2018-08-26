#ifndef SIMULATION_H
#define SIMULATION_H

#include <array>
#include <cstdio>
#include <stdexcept>
#include <unistd.h>
#include <vector>

#include "RandomGenerators.h"
#include "Step.h"

class Simulation {
  Step *step;
  unsigned numberOfBodies;
  thrust::host_vector<float> positions;
  RandomGenerators *randomGenerator;

public:
  Simulation(Step *step, unsigned numberOfBodies);
  ~Simulation();
  void MakeSimulation(int rounds);
};

#endif
