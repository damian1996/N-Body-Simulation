#include "Simulation.h"
#include <chrono>

Simulation::Simulation(Step *step, unsigned numberOfBodies)
    : step(step), numberOfBodies(numberOfBodies) {
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
}

void Simulation::MakeSimulation(int rounds) {
  typedef std::chrono::high_resolution_clock Time;
  typedef std::chrono::duration<float> fsec;
  fsec fs;
  auto last_time = Time::now();
  auto before_time = Time::now();
  int count = 0;
  int numberOfRounds = rounds;
  while(1) {
      if(count >= numberOfRounds) break;
      auto curr_time = Time::now();
      fs = curr_time - last_time;
      float dt = fs.count();
      step->compute(positions, dt);
      last_time = curr_time;
      count++;
  }
  auto after_time = Time::now();
  fsec timeInSec = after_time - before_time;
  float avgDt = timeInSec.count();
  float res = avgDt/numberOfRounds;
  std::cout << "(" << numberOfRounds << "," << res << ")" << std::endl;
}