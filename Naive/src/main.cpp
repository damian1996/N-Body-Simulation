#include "main.h"
using namespace std;

int main() {
    // dodac wczytanie N przez usera, poki co roboczo 100
    int type, N;
    printf("Wybierz tryb wykonywania programu\n");
    printf("1. Naiwny algorytm CPU\n2. Naiwny algorytm GPU\n");
    scanf("%d", &type);
    printf("Podaj liczbę jednostek do poddania symulacji\n");
    scanf("%d", &N);
    Simulation sim(N);
    switch(type) {
      case 1:
      {
        sim.makeSimulation();
        break;
      }
      case 2:
      {
        sim.makeSimulationCUDA();
        break;
      }
      default:
        break;
    }
    return 0;
}
