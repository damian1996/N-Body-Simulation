#include "StepNaive.h"

StepNaive::StepNaive(unsigned N) {
    this->N = N;
    forces.resize(3*N);
    velocities.resize(3*N);
    weights.resize(N);
    rg = new RandomGenerators();
    for(unsigned i=0; i<N; i++) {
        for(int j=0; j<3; j++) {
            velocities[i*3+j] = rg->getRandomFloat(-0.01f, 0.01f);
        }
        weights[i] = rg->getRandomFloat(10.0f, 10005.0f);
    }
}

StepNaive::~StepNaive() {
  delete rg;
  weights.clear();
  velocities.clear();
  forces.clear();
}

void StepNaive::compute(tf3& positions, float dt) {
    std::fill(forces.begin(), forces.end(), 0);
    for(unsigned i=0; i<N; i++) {
        for(unsigned j=0; j<N; j++) {
            float distX = positions[j*3] - positions[i*3];
            float distY = positions[j*3+1] - positions[i*3+1];
            if(i!=j && fabs(distX) > 1e-10 && fabs(distY) > 1e-10) {
                float F = G*(weights[i]*weights[j]);
                forces[i*3] += F*distX/(distX*distX+distY*distY); // force = G(m1*m2)/r^2
                forces[i*3+1] += F*distY/(distX*distX+distY*distY);
            }
        }
    }
    for(unsigned i=0; i<N; i++) {
        for(int j=0; j<2; j++) { // x, y
            float acceleration = forces[i*3+j]/weights[i];
            positions[i*3+j] += velocities[i*3+j]*dt + acceleration*dt*dt/2;
            velocities[i*3+j] += acceleration*dt;
        }
    }
}

void StepNaive::initialize() {

}
