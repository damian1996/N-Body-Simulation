#ifndef RANDOMGENERATORS_H
#define RANDOMGENERATORS_H

#include <random>
#include <thrust/host_vector.h>
#include <vector>

class RandomGenerators {
public:
  RandomGenerators();
  float getRandomFloat(float a, float b);
  int getRandomColor();
  int getRandomType();
  template <typename T>
  void initializeVelocities(T &velocities, unsigned numberOfBodies);

  void initializeWeights(std::vector<float>& weights, unsigned numberOfBodies);
};

#endif
