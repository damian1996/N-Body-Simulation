#include "Main.h"
using namespace std;

int main() {
    // dodac wczytanie N przez usera, poki co roboczo 100
    int type, N;
    printf("Wybierz tryb wykonywania programu\n");
    printf("1. Naiwny algorytm CPU\n2. Naiwny algorytm GPU\n3. Algorytm KD-drzewa CPU\n4. Algorytm Barnes-Hut GPU\n");
    scanf("%d", &type);
    printf("Podaj liczbę jednostek do poddania symulacji\n");
    scanf("%d", &N);
    printf("WTF... %d\n", N);
    RandomGenerators* ran_gen = new RandomGenerators();
    std::vector<float> masses(N);
    ran_gen->initializeWeights<std::vector<float>>(masses, N);
    Render* r = new Render(masses, N);
    Step* step;
    Simulation* sim;

    while(1) {
        bool correctChoice = true;
        switch(type) {
            case 1:
            {
              step = new StepNaive(masses, N);
              sim = new Simulation(r, step, N);
              sim->makeSimulation();
              break;
            }
            case 2:
            {
              step = new StepNaiveCuda(masses, N);
              sim = new Simulation(r, step, N);
              sim->makeSimulation();
              break;
            }
            case 3:
            {
              printf("Implementacja wciaz nie powstala, troche cierpliwosci :)\n");
              break;
            }
            case 4:
            {
              printf("Implementacja wciaz nie powstala, troche cierpliwosci :)\n");
              break;
            }
            default:
              correctChoice = false;
              break;
        }
        if(correctChoice) break;
    }

    delete ran_gen;
    return 0;
}

// https://pl.wikipedia.org/wiki/Wstrzykiwanie_zale%C5%BCno%C5%9Bci
// https://arxiv.org/pdf/0806.3950.pdf

//file:///home/damian/Desktop/praca%20licencjacka/materialy/Gravitational%20N-Body%20Simulatio%20-%20Aarseth,%20Sverre%20J._5510.pdf
//file:///home/damian/Desktop/praca%20licencjacka/materialy/nBody_nVidia.pdf
//file:///home/damian/Desktop/praca%20licencjacka/materialy/Gravitational%20N-Body%20Simulatio%20-%20Aarseth,%20Sverre%20J._5510.pdf
//https://www.sciencedirect.com/science/article/pii/0021999173901605
//https://github.com/adityavkk/N-Body-Simulations/blob/master/Barnes-Hut/src/Bodies.hs
//https://link.springer.com/chapter/10.1007%2F978-1-4613-9600-0_7
//https://www.maths.tcd.ie/~btyrrel/nbody.pdf
