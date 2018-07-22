#include "StepNaiveCUDA.h"

// lista inicjalizacyjna vs dziedziczenie?

StepNaiveCuda::StepNaiveCuda(std::vector<float> masses, unsigned N) {
    this->N = N;
    weights.resize(N);
    for(unsigned i=0; i<N; i++) {
        weights[i] = masses[i];
    }
    rg = new RandomGenerators();
    rg->initializeVelocities<thrust::host_vector<float> >(velocities, N);
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
