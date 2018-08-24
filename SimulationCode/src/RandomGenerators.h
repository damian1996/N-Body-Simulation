#ifndef RANDOMGENERATORS_H
#define RANDOMGENERATORS_H

#include <random>
#include <thrust/host_vector.h>
#include <vector>

class RandomGenerators {
  std::random_device rd1;
  std::mt19937 gen1;
  std::random_device rd2;
  std::mt19937 gen2;
public:
  RandomGenerators();
  double getRandomFloat(double a, double b);
  int getRandomColor();
  int getRandomType();
  template <typename T>
  void initializeVelocities(T &velocities, unsigned numberOfBodies);

  void initializeWeights(std::vector<double>& weights, unsigned numberOfBodies);
};

#endif
