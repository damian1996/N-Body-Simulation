#ifndef SIMULATION_H
#define SIMULATION_H

#include <array>
#include <cstdio>
#include <stdexcept>
#include <unistd.h>
#include <vector>
#include <cmath>
#include <thread>
#include <chrono>

#include "gl.h"
#include "render.h"
#include "RandomGenerators.h"
#include "Simulation.cuh"

class Simulation {

    unsigned N;
    const float G = 6.674*(1e-10);
    //const float G = 6.674*(1e-11); // 6.674×10−11 m3⋅kg−1⋅s−2
    const float dt = 0.016f;
    Scene* painter;
    RandomGenerators rg;
    tf3 weights;
    tf3 positions;
    tf3 velocities;
    std::vector<std::array<char, 3> > colors;
    std::vector<std::array<float, 3> > forces;
    //std::vector<std::array<float, 3> > positions;
    //std::vector<float> weights;
    //std::vector<std::array<float, 3> > velocities;
public:
    Simulation();
    Simulation(unsigned N);
    ~Simulation();
    void randomData();
    bool update();
    void makeSimulation();
    void makeSimulationCUDA();
};

#endif
