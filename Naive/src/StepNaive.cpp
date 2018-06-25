#include "StepNaive.h"

StepNaive::StepNaive(unsigned N) {
    this->N = N;
    forces.resize(3*N);
    rg = new RandomGenerators();
    rg->initializeValues<std::vector<float> >(velocities, weights, N);
}

StepNaive::~StepNaive() {
  delete rg;
  weights.clear();
  velocities.clear();
  forces.clear();
}

void StepNaive::compute(tf3& positions, float dt) {
    /*
    float zzpx = 0.0f, zzpy = 0.0f;
    for(int i=0; i<N; i++) {
        zzpx += (weights[i]*velocities[i*3]);
        zzpy += (weights[i]*velocities[i*3+1]);
    }
    std::cout << "Pedy : " << zzpx << "  " << zzpy << std::endl;
    */
    float EPS = 0.001f;
    std::fill(forces.begin(), forces.end(), 0);
    for(unsigned i=0; i<N; i++) {
        for(unsigned j=0; j<N; j++) {
            float distX = positions[j*3] - positions[i*3];
            float distY = positions[j*3+1] - positions[i*3+1];
            float d = (distX*distX+distY*distY) + EPS*EPS;
            if(i!=j && fabs(distX) > 1e-10 && fabs(distY) > 1e-10) {
                  float F = G*(weights[i]*weights[j]);
                  forces[i*3] += F*distX/d; // force = G(m1*m2)/r^2
                  forces[i*3+1] += F*distY/d;
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
