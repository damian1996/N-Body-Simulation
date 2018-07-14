#ifndef RANDOMGENERATORS_H
#define RANDOMGENERATORS_H

#include <random>
#include <vector>
#include <thrust/host_vector.h>

class RandomGenerators {
public:
  RandomGenerators();
  double getRandomdouble(double a, double b);
  //double randPosition(double a, double b);
  int getRandomByte();
  int getRandomType();
  template<typename T>
  void initializeVelocities(T& velocities, unsigned N);
  
  template<typename T>
  void initializeWeights(T& weights, unsigned N);
};

#endif
