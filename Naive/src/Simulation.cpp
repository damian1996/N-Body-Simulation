#include "Simulation.h"

Simulation::Simulation(Render *rend, Step *step, unsigned numberOfBodies)
    : rend(rend), step(step), numberOfBodies(numberOfBodies) {
  randomGenerator = new RandomGenerators();
  positions.reserve(3 * numberOfBodies);
  for (unsigned i = 0; i < numberOfBodies; i++) {
    for (int j = 0; j < 3; j++) {
       float randPos = randomGenerator->getRandomFloat(-1.0f, 1.0f);
       positions.push_back(randPos);
       //std::cout << randPos << std::endl;
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

  while(!rend->draw(positions)) {
      curr_time = rend->getTime();
      step->compute(positions, (curr_time - last_time) * 10);
      last_time = curr_time;
  }
}
