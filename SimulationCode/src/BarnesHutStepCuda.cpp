#include "BarnesHutStepCuda.h"

BarnesHutStepCuda::BarnesHutStepCuda(std::vector<float>& masses, unsigned numberOfBodies) {
    this->numberOfBodies = numberOfBodies;
    weights.resize(numberOfBodies);
    forces.resize(3*numberOfBodies);
    weights = std::vector<float>(masses.begin(), masses.end());
    randomGenerator = new RandomGenerators();
    randomGenerator->initializeVelocities<std::vector<float>>(velocities, numberOfBodies);
    c = new ComputationsBarnesHut(velocities, weights);
}

BarnesHutStepCuda::~BarnesHutStepCuda() {
    delete c;
    delete randomGenerator;
}

void BarnesHutStepCuda::compute(tf3 &positions, float dt) {
  c->BarnesHutBridge(positions, numberOfBodies, dt);
}
