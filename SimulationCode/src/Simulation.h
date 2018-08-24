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
  Render *rend;
  Step *step;
  unsigned numberOfBodies;
  thrust::host_vector<double> positions;
  RandomGenerators *randomGenerator;

public:
  Simulation(Render *rend, Step *step, unsigned numberOfBodies);
  ~Simulation();
  void MakeSimulation();
};

#endif
