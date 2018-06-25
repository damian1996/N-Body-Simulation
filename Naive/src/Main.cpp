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
    Render* r = new Render(N);
    Step* step;
    Simulation* sim;

    

    while(1) {
        bool correctChoice = true;
        switch(type) {
            case 1:
            {
              step = new StepNaive(N);
              sim = new Simulation(r, step, N);
              sim->makeSimulation();
              break;
            }
            case 2:
            {
              step = new StepNaiveCuda(N);
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

    return 0;
}

// https://pl.wikipedia.org/wiki/Wstrzykiwanie_zale%C5%BCno%C5%9Bci
// https://arxiv.org/pdf/0806.3950.pdf
