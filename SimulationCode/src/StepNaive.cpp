#include "StepNaive.h"

StepNaive::StepNaive(std::vector<float> &masses, unsigned numberOfBodies) {
  firstStep = false;
  this->numberOfBodies = numberOfBodies;
  forces.resize(3 * numberOfBodies);
  weights.resize(numberOfBodies);
  for (unsigned i = 0; i < numberOfBodies; i++)
    weights[i] = masses[i];
  randomGenerator = new RandomGenerators();
  randomGenerator->initializeVelocities<std::vector<float>>(velocities, numberOfBodies);
}

StepNaive::~StepNaive() {
  delete randomGenerator;
  weights.clear();
  velocities.clear();
  forces.clear();
}

bool StepNaive::testingMomemntum() {
    float momentumX = 0.0f, momentumY = 0.0f, momentumZ = 0.0f;
    for (unsigned i = 0; i < numberOfBodies; i++) {
        momentumX += (weights[i] * velocities[i * 3]);
        momentumY += (weights[i] * velocities[i * 3 + 1]);
        momentumZ += (weights[i] * velocities[i * 3 + 2]);
    }
    if(!firstStep) {
      firstStep = true;
      oldMomentumX = momentumX;
      oldMomentumY = momentumY;
      oldMomentumZ = momentumZ;
    }

    if(fabs(oldMomentumX - momentumX) > 1.0) return false;
    if(fabs(oldMomentumY - momentumY) > 1.0) return false;
    if(fabs(oldMomentumZ - momentumZ) > 1.0) return false;

    oldMomentumX = momentumX;
    oldMomentumY = momentumY;
    oldMomentumZ = momentumZ;
    return true;
}

void StepNaive::compute(tf3 &positions, float dt) {
  //assert(testingMomemntum());
  float EPS = 0.01;
  std::fill(forces.begin(), forces.end(), 0.0);
  for (unsigned i = 0; i < numberOfBodies; i++) {
    for (unsigned j = 0; j < numberOfBodies; j++) {
      float distX = positions[j * 3] - positions[i * 3];
      float distY = positions[j * 3 + 1] - positions[i * 3 + 1];
      float distZ = positions[j * 3 + 2] - positions[i * 3 + 2];
      float dist = (distX * distX + distY * distY + distZ * distZ) + EPS * EPS;
      dist = dist * sqrt(dist);
      if (i != j) {
        float F = G * (weights[i] * weights[j]);
        forces[i * 3] += F * distX / dist; // force = G(m1*m2)/r^2
        forces[i * 3 + 1] += F * distY / dist;
        forces[i * 3 + 2] += F * distZ / dist;
      }
    }
  }
  for (unsigned i = 0; i < numberOfBodies; i++) {
    for (int j = 0; j < 3; j++) {
      float acceleration = forces[i * 3 + j] / weights[i];
      positions[i * 3 + j] +=
          velocities[i * 3 + j] * dt + acceleration * dt * dt / 2;
      velocities[i * 3 + j] += acceleration * dt;
    }
  }
}
