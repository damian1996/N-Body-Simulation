#include "Main.h"
using namespace std;

int main() {
  // dodac wczytanie N przez usera, poki co roboczo 100

  int programVersion, numberOfBodies;
  printf("Wybierz tryb wykonywania programu\n");
  while (1) {
    printf("1. Naiwny algorytm CPU\n2. Naiwny algorytm GPU\n3. Algorytm "
           "KD-drzewa CPU\n4. Algorytm Barnes-Hut GPU\n");
    scanf("%d", &programVersion);
    if (programVersion > 4 || programVersion <= 0) {
      printf("Nie ma takiej opcji, sprobuj ponownie\n");
      continue;
    }
    break;
  }
  printf("Podaj liczbę jednostek do poddania symulacji\n");
  scanf("%d", &numberOfBodies);
  RandomGenerators *randomGenerator = new RandomGenerators();
  std::vector<float> masses(numberOfBodies);
  randomGenerator->initializeWeights(masses, numberOfBodies);
  Render *rend = new Render(masses, numberOfBodies);
  Step *step;
  Simulation *sim;

  switch (programVersion) {
    case 1: {
      step = new StepNaive(masses, numberOfBodies);
      break;
    }
    case 2: {
      step = new StepNaiveCuda(masses, numberOfBodies);
      break;
    }
    case 3: {
      step = new BarnesHutStep(masses, numberOfBodies);
      break;
    }
    case 4: {
      printf("Implementacja wciaz nie powstala, troche cierpliwosci :)\n");
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

// https://pl.wikipedia.orandomGenerator/wiki/Wstrzykiwanie_zale%C5%BCno%C5%9Bci
// https://arxiv.orandomGenerator/pdf/0806.3950.pdf

// file:///home/damian/Desktop/praca%20licencjacka/materialy/Gravitational%20N-Body%20Simulatio%20-%20Aarseth,%20Sverre%20J._5510.pdf
// file:///home/damian/Desktop/praca%20licencjacka/materialy/nBody_nVidia.pdf
// file:///home/damian/Desktop/praca%20licencjacka/materialy/Gravitational%20N-Body%20Simulatio%20-%20Aarseth,%20Sverre%20J._5510.pdf
// https://www.sciencedirect.com/science/article/pii/0021999173901605
// https://github.com/adityavkk/N-Body-Simulations/blob/master/Barnes-Hut/src/Bodies.hs
// https://link.springer.com/chapter/10.1007%2F978-1-4613-9600-0_7
// https://www.maths.tcd.ie/~btyrrel/nbody.pdf
