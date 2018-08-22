#include "BarnesHutStepCuda.h"

BarnesHutStepCuda::BarnesHutStepCuda(std::vector<float>& masses, unsigned numberOfBodies) {
    this->numberOfBodies = numberOfBodies;
    weights.resize(numberOfBodies);
    forces.resize(3*numberOfBodies);

    //std::array<double, 6> boundariesForRoot = {-sizeFrame, sizeFrame, -sizeFrame, sizeFrame, -sizeFrame, sizeFrame};
    //root = new NodeBH(boundariesForRoot);

    weights = std::vector<double>(masses.begin(), masses.end());

    randomGenerator = new RandomGenerators();
    randomGenerator->initializeVelocities<std::vector<double>>(velocities, numberOfBodies);

    c = new ComputationsBarnesHut(velocities, weights);
}

BarnesHutStepCuda::~BarnesHutStepCuda() {
    delete c;
    delete randomGenerator;
}

void BarnesHutStepCuda::compute(tf3 &positions, float dt) {
  c->BarnesHutBridge(positions, numberOfBodies, dt);
}
