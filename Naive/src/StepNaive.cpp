#include "StepNaive.h"

StepNaive::StepNaive(std::vector<float>& masses, unsigned N) {
    this->N = N;
    forces.resize(3*N);
    weights.resize(N);
    for(unsigned i=0; i<N; i++) weights[i] = masses[i];
    rg = new RandomGenerators();
    rg->initializeVelocities<std::vector<float> >(velocities, N);
}

StepNaive::~StepNaive() {
    delete rg;
    weights.clear();
    velocities.clear();
    forces.clear();
}

void StepNaive::compute(tf3& positions, float dt) {
    float zzpx = 0.0f, zzpy = 0.0f, zzpz = 0.0f;
    for(unsigned i=0; i<N; i++) {
        zzpx += (weights[i]*velocities[i*3]);
        zzpy += (weights[i]*velocities[i*3+1]);
        zzpz += (weights[i]*velocities[i*3+2]);
    }
    // std::cout << "Pedy : " << zzpx << "  " << zzpy << " " << zzpz << std::endl;
    float EPS = 0.001f;
    std::fill(forces.begin(), forces.end(), 0);
    for(unsigned i=0; i<N; i++) {
        for(unsigned j=0; j<N; j++) {
            float distX = positions[j*3] - positions[i*3];
            float distY = positions[j*3+1] - positions[i*3+1];
            float distZ = positions[j*3+2] - positions[i*3+2];
            float dist = (distX*distX + distY*distY + distZ*distZ) + EPS*EPS;
            float distSqrt = sqrt(dist);
            dist = dist*distSqrt;
            if(i==0 && j==5) {
                // std::cout <<  positions[j*3] << "  " << positions[i*3] << std::endl;
                // std::cout <<  positions[j*3 + 2] << "  " << positions[i*3 + 2] << std::endl;
                // std::cout <<  distX << "  " << distZ << std::endl;
                // std::cout <<  dist  << std::endl;
            }
            if(i!=j) { // && fabs(distX) > 1e-10 && fabs(distY) > 1e-10 && fabs(distZ) > 1e-10) {
                float F = G*(weights[i]*weights[j]);
                if(i==0 && j==5) {
                    //std::cout << "--- " << F << std::endl;
                    //std::cout << weights[i] << " " << weights[j] << std::endl;
                } 
                forces[i*3] += F*distX/dist; // force = G(m1*m2)/r^2
                forces[i*3+1] += F*distY/dist;
                forces[i*3+2] += F*distZ/dist;
            }
        }
    }
    for(unsigned i=0; i<N; i++) {
        for(int j=0; j<3; j++) { // x, y
            float acceleration = forces[i*3+j]/weights[i];
            positions[i*3+j] += velocities[i*3+j]*dt + acceleration*dt*dt/2;
            if(i==5) {
                // std::cout << "at dimension " << j << " position is " << positions[i*3 + j] << std::endl;
            }
            velocities[i*3+j] += acceleration*dt;
        }
    }
}

void StepNaive::initialize() {

}