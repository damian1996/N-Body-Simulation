#include "StepNaiveCUDA.h"

// lista inicjalizacyjna vs dziedziczenie?

StepNaiveCuda::StepNaiveCuda(unsigned N) {
    this->N = N;
    rg = new RandomGenerators();
    velocities.resize(3*N);
    weights.resize(N);
    for(unsigned i=0; i<N; i++) {
        for(int j=0; j<3; j++)
            velocities[i*3+j] = rg->getRandomfloat(-0.01f, 0.01f);
        weights[i] = rg->getRandomfloat(1000.0f, 100000.0f);
    }
    c = new Computations(velocities, weights);
}

StepNaiveCuda::~StepNaiveCuda() {
    delete c;
    weights.clear();
    velocities.clear();
    delete rg;
}

void StepNaiveCuda::compute(tf3& positions, float dt) {
    c->NaiveSimBridgeThrust(positions, N, dt);
}
