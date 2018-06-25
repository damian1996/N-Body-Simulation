#ifndef RANDOMGENERATORS_H
#define RANDOMGENERATORS_H

#include <random>
#include <vector>
#include <thrust/host_vector.h>

class RandomGenerators {
public:
  RandomGenerators();
  float getRandomfloat(float a, float b);
  //float randPosition(float a, float b);
  int getRandomByte();
  int getRandomType();
  template<typename T>
  void initializeValues(T& velocities, T& weights, unsigned N);
};

#endif
