#include "StepNaiveCUDA.h"

// lista inicjalizacyjna vs dziedziczenie?

StepNaiveCuda::StepNaiveCuda(unsigned N) {
    this->N = N;
    rg = new RandomGenerators();
    rg->initializeValues<thrust::host_vector<float> >(velocities, weights, N);

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
