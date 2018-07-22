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
  thrust::host_vector<double> positions;
  thrust::host_vector<double> posToDraw;
  RandomGenerators *rg;
  const double parsecsWidth = 1.0;

public:
  Simulation(Render *r, Step *comp, unsigned N);
  ~Simulation();
  void normalizePositions();
  void makeSimulation();
};

#endif
