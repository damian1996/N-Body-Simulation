#include "StepNaiveCUDA.h"

// lista inicjalizacyjna vs dziedziczenie?

StepNaiveCuda::StepNaiveCuda(std::vector<double> masses, unsigned N) {
  this->N = N;
  weights.resize(N);
  for (unsigned i = 0; i < N; i++) {
    weights[i] = masses[i];
  }
  rg = new RandomGenerators();
  rg->initializeVelocities<thrust::host_vector<double>>(velocities, N);
  c = new Computations(velocities, weights);
}

StepNaiveCuda::~StepNaiveCuda() {
  delete c;
  weights.clear();
  velocities.clear();
  delete rg;
}

void StepNaiveCuda::compute(tf3 &positions, double dt) {
  c->NaiveSimBridgeThrust(positions, N, dt);
}
