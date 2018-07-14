#include "Simulation.h"

Simulation::Simulation(Render* r, Step* comp, unsigned N) : r(r), comp(comp), N(N){
    rg = new RandomGenerators();
    positions.reserve(3*N);
    for(unsigned i=0; i<N; i++) {
        for(int j=0; j<3; j++)
            positions.push_back(rg->getRandomfloat(-1.0f, 1.0f));
        //positions.push_back(0.0f);
    }
    posToDraw.resize(3*N);
}

Simulation::~Simulation() {
    delete rg;
    positions.clear();
    delete comp;
    delete r;
}

void Simulation::normalizePositions() {
    for(int i=0; i < 3*N; i++) {
      posToDraw[i] = positions[i]; // /200.0;
    }
}

void Simulation::makeSimulation() {
    r->setupOpenGL();
    double last_time = r->getTime();
    double curr_time;
    normalizePositions();
    r->draw(posToDraw);
    while(true) {
        curr_time = r->getTime();
        //printf("%f %f\n", curr_time, 10*(curr_time-last_time));
        comp->compute(positions, (curr_time-last_time)*10);//100*(curr_time-last_time));
        last_time = curr_time;
        normalizePositions();
        bool v = r->draw(posToDraw);
        //bool v = r->draw(positions);
        if(v) break;
    }
}

// src/Simulation.cpp:14:12: warning: deleting object of abstract class type ‘Step’ which has non-virtual
// destructor will cause undefined behaviour [-Wdelete-non-virtual-dtor] -> delete comp;
