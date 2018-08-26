#include "Main.h"
using namespace std;

int main() {
  int programVersion, numberOfBodies;
  printf("Wybierz tryb wykonywania programu\n");
  while (1) {
    printf("1. Naiwny algorytm CPU\n2. Naiwny algorytm GPU\n3. Algorytm "
           "Barnes-Hut CPU\n4. Algorytm Barnes-Hut GPU\n");
    int res = scanf("%d", &programVersion);
    if (programVersion > 4 || programVersion <= 0) {
      printf("Nie ma takiej opcji, sprobuj ponownie\n");
      continue;
    }
    break;
  }
  printf("Podaj liczbę jednostek do poddania symulacji\n");
  int res = scanf("%d", &numberOfBodies);
  if(res == EOF) return 1;
  RandomGenerators *randomGenerator = new RandomGenerators();
  std::vector<float> masses(numberOfBodies);
  randomGenerator->initializeWeights(masses, numberOfBodies);
  Render *rend = nullptr; 
  Step *step = nullptr;
  Simulation *sim;

  switch (programVersion) {
    case 1: {
      rend = new Render(masses, numberOfBodies, 1.0);
      step = new StepNaive(masses, numberOfBodies);
      break;
    }
    case 2: {
      rend = new Render(masses, numberOfBodies, 1.0);
      step = new StepNaiveCuda(masses, numberOfBodies);
      break;
    }
    case 3: {
      rend = new Render(masses, numberOfBodies, 4.0);
      step = new BarnesHutStep(masses, numberOfBodies);
      break;
    }
    case 4: {
      rend = new Render(masses, numberOfBodies, 4.0);
      step = new BarnesHutStepCuda(masses, numberOfBodies);
      break;
    }
    default:
      break;
  }
  sim = new Simulation(rend, step, numberOfBodies);
  sim->MakeSimulation();

  delete randomGenerator;
  return 0;
}
