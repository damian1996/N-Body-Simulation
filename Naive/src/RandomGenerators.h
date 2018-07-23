#ifndef RANDOMGENERATORS_H
#define RANDOMGENERATORS_H

#include <random>
#include <thrust/host_vector.h>
#include <vector>

class RandomGenerators {
public:
  RandomGenerators();
  float getRandomfloat(float a, float b);
  // float randPosition(float a, float b);
  int getRandomByte();
  int getRandomType();
  template <typename T> void initializeVelocities(T &velocities, unsigned N);

  template <typename T> void initializeWeights(T &weights, unsigned N);
};

#endif
