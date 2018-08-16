#include "Simulation.h"

Simulation::Simulation(Render *rend, Step *step, unsigned numberOfBodies)
    : rend(rend), step(step), numberOfBodies(numberOfBodies) {
  randomGenerator = new RandomGenerators();
  positions.reserve(3 * numberOfBodies);
  for (unsigned i = 0; i < numberOfBodies; i++) {
    for (int j = 0; j < 3; j++) {
       float randPos = randomGenerator->getRandomFloat(-1.0f, 1.0f);
       positions.push_back(randPos);
    }
  }
}

Simulation::~Simulation() {
  delete randomGenerator;
  positions.clear();
  delete step;
  delete rend;
}

void Simulation::MakeSimulation() {
  rend->setupOpenGL();
  float last_time = rend->getTime();
  float curr_time;

  int count = 0;
  float avgDt = 0.0;
  while(!rend->draw(positions)) {
      if(count > 100) break;
      curr_time = rend->getTime();
      //std::cout << curr_time - last_time << std::endl;
      avgDt += (curr_time - last_time);
      step->compute(positions, (curr_time - last_time));// * 10);
      last_time = curr_time;
      count++;
  }
  std::cout << avgDt/1000 << std::endl;
}
