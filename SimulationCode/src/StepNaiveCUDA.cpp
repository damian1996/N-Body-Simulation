#include "StepNaiveCUDA.h"

StepNaiveCuda::StepNaiveCuda(std::vector<float> masses, unsigned numberOfBodies) {
  this->numberOfBodies = numberOfBodies;
  weights.resize(numberOfBodies);
  for (unsigned i = 0; i < numberOfBodies; i++) {
    weights[i] = masses[i];
  }
  randomGenerator = new RandomGenerators();
  randomGenerator->initializeVelocities<thrust::host_vector<float>>(velocities, numberOfBodies);
  c = new Computations(velocities, weights);
}

StepNaiveCuda::~StepNaiveCuda() {
  delete c;
  weights.clear();
  velocities.clear();
  delete randomGenerator;
}

void StepNaiveCuda::compute(tf3 &positions, float dt) {
  c->NaiveSimBridgeThrust(positions, numberOfBodies, dt);
}
