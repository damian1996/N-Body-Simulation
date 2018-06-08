#include "Simulation.h"

Simulation::Simulation(Render* r, Step* comp, unsigned N) : r(r), comp(comp), N(N){
    rg = new RandomGenerators();
    positions.reserve(3*N);
    for(unsigned i=0; i<N; i++) {
        for(int j=0; j<2; j++)
            positions.push_back(rg->getRandomFloat(-1.0, 1.0));
        positions.push_back(0.0f);
    }
}

Simulation::~Simulation() {
    delete rg;
    positions.clear();
    delete comp;
    delete r;
}

void Simulation::makeSimulation() {
    r->setupOpenGL();
    float last_time = r->getTime();
    float curr_time;
    r->draw(positions);
    while(true) {
        curr_time = r->getTime();
        //printf("%f %f\n", curr_time, 10*(curr_time-last_time));
        comp->compute(positions, 10*(curr_time-last_time));
        last_time = curr_time;
        bool v = r->draw(positions);
        if(v) break;
    }
}

// src/Simulation.cpp:14:12: warning: deleting object of abstract class type ‘Step’ which has non-virtual
// destructor will cause undefined behaviour [-Wdelete-non-virtual-dtor] -> delete comp;
