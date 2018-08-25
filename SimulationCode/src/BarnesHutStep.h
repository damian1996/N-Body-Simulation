#ifndef BARNESHUTSTEP_H
#define BARNESHUTSTEP_H

#include "Step.h"
#include "NodeBH.h"
#include "RandomGenerators.h"
#include <thrust/host_vector.h>
#include <algorithm>
#include <array>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

class BarnesHutStep : public Step {
public:
    BarnesHutStep(std::vector<float>& masses, unsigned numberOfBodies);
    ~BarnesHutStep();
    void initializingRoot();
    void insertNode(NodeBH* node, NodeBH* quad);
    void createTree(tf3 &positions);
    void DFS_BH(NodeBH* r, std::string indent);
    void cleanUpTreePostOrder(NodeBH* r);
    float dist(float one, float two);
    float distanceBetweenTwoNodes(float x1, float y1, float z1, float x2, float y2, float z2);
    void testingMomemntum();
    void computeForceForBody(NodeBH* r, std::array<float, 3>& pos, int i);
    void compute(tf3 &positions, float dt);

private:
    NodeBH* root;
    std::vector<float> weights;
    std::vector<float> forces;
    std::vector<float> velocities;
    float sizeFrame = 8.0; // ZWIEKSZYC (DO 32.0)
    const float theta = 0.5;
    const float EPS = 0.01;
};

#endif

// https://stackoverflow.com/questions/41946007/efficient-and-well-explained-implementation-of-a-quadtree-for-2d-collision-det
// http://www.cs.umd.edu/~hjs/pubs/ShaffCVGIP87.pdf
// http://arborjs.org/docs/barnes-hut
// https://www.khanacademy.org/science/physics/linear-momentum/center-of-mass/a/what-is-center-of-mass
// http://iss.ices.utexas.edu/Publications/Papers/burtscher11.pdf
// https://bitbucket.org/jsandham/nbodycuda/overview
// https://en.wikipedia.org/wiki/Octree#Application_to_color_quantization
// https://scala-blitz.github.io/home/documentation/examples//barneshut.html
// https://github.com/chindesaurus/BarnesHut-N-Body/blob/master/NBodyBH.java
