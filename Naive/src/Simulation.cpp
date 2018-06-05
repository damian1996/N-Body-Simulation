#include "Simulation.h"

Simulation::Simulation() : N(10){
  printf("AJA\n");
  painter = new Scene(N);
  positions.resize(3*N);
  velocities.resize(3*N);
  weights.resize(N);
  colors.resize(N);
  forces.resize(N);
}

Simulation::Simulation(unsigned N) : N(N){
  painter = new Scene(N);
  //positions.reserve(3*N);
  positions.resize(3*N);

  //velocities.reserve(3*N);
  velocities.resize(3*N);

  //weights.reserve(N);
  weights.resize(N);

  colors.resize(N);
  forces.resize(N);
}

Simulation::~Simulation() {
    forces.clear();
    colors.clear();
    weights.clear();
    velocities.clear();
    positions.clear();
    delete painter;
}

void Simulation::randomData() {
    for(unsigned i=0; i<N; i++) {
        for(int j=0; j<2; j++)
            positions[i*3+j] = rg.getRandomFloat(-1.0, 1.0);
        positions[i*3+2] = 0.0f;

        for(int j=0; j<3; j++)
            velocities[i*3+j] = 0.0f;

        for(int j=0; j<3; j++)
            colors[i][j] = rg.getRandomByte();

        weights[i] = rg.getRandomFloat(10.0f, 10005.0f);
    }
}

bool Simulation::update() {
        for(unsigned i=0; i<N; i++) {
            forces[i].fill(0.0f);
        }
        for(unsigned i=0; i<N; i++) {
            for(unsigned j=0; j<N; j++) {
                float distX = positions[j*3] - positions[i*3];
                float distY = positions[j*3+1] - positions[i*3+1];
                if(i!=j && fabs(distX) > 1e-10 && fabs(distY) > 1e-10) {
                    float F = G*(weights[i]*weights[j]);
                    forces[i][0] += F*distX/(distX*distX+distY*distY); // force = G(m1*m2)/r^2
                    forces[i][1] += F*distY/(distX*distX+distY*distY);
                }
            }
        }
        for(unsigned i=0; i<N; i++) {
            for(int j=0; j<2; j++) { // x, y
                float acceleration = forces[i][j]/weights[i];
                positions[i*3+j] += velocities[i*3+j]*dt + acceleration*dt*dt/2;
                velocities[i*3+j] += acceleration*dt;
            }
        }
        painter->sendData(positions);
        return painter->draw();
}

void Simulation::makeSimulation() {
    randomData();
    painter->sendInitialData(positions, colors, weights);
    painter->setupOpenGL();
    while(true) {
        bool v = update();
        if(v) break;
    }
}

void Simulation::makeSimulationCUDA() {
    randomData();
    painter->sendInitialData(positions, colors, weights);
    painter->setupOpenGL();
    NaiveSimBridgeThrust(painter, positions, velocities, weights, N);
}
