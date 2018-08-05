#ifndef BARNESHUTSTEP_H
#define BARNESHUTSTEP_H

#include "Step.h"
#include "NodeBH.h"
#include "RandomGenerators.h"
#include <thrust/host_vector.h>
#include <iterator>
#include <vector>
#include <algorithm>
#include <iostream>

typedef thrust::host_vector<NodeBH*> tn3;

class BarnesHutStep : public Step {
public:
    BarnesHutStep(std::vector<float>& masses, unsigned numberOfBodies);
    ~BarnesHutStep();
    bool isCreate() const;
    void setCreate();
    void insertNode(NodeBH* node, NodeBH* quad);
    void createTree(tf3 &positions);
    void DFS_BH(NodeBH* r); // Debug
    double dist(double one, double two);
    double distanceBetweenTwoNodes(double x1, double y1, double z1, double x2, double y2, double z2);
    void computeForceForBody(NodeBH* r, std::vector<double> pos, int i);
    // void computeForceForBody(NodeBH* r, double x1, double x2, int i, int j);
    void compute(tf3 &positions, float dt);

private:
    NodeBH* root;
    bool wasCreated = false;
    std::vector<double> weights;
    std::vector<double> forces;
    std::vector<double> velocities;
    double sizeFrame = 2.0; // ZWIEKSZYC W CIUL (DO 32.0)
    const int numberOfQuads = 8;
    const double theta = 0.5;
    const double EPS = 0.01;
    int counter = 0; // Debug
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
