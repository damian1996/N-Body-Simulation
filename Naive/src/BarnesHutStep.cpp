#include "BarnesHutStep.h"

BarnesHutStep::BarnesHutStep(std::vector<float>& masses, unsigned numberOfBodies) {
    this->numberOfBodies = numberOfBodies;
    weights.resize(numberOfBodies);
    forces.resize(3*numberOfBodies);

    std::array<double, 6> boundariesForRoot = {-sizeFrame, sizeFrame, -sizeFrame, sizeFrame, -sizeFrame, sizeFrame};
    root = new NodeBH(boundariesForRoot);

    weights = std::vector<double>(masses.begin(), masses.end());

    randomGenerator = new RandomGenerators();
    randomGenerator->initializeVelocities<std::vector<double>>(velocities, numberOfBodies);
}

BarnesHutStep::~BarnesHutStep() {
    delete randomGenerator;
}

void BarnesHutStep::initializingRoot() {
    std::array<double, 6> boundariesForRoot = {-sizeFrame, sizeFrame, -sizeFrame, sizeFrame, -sizeFrame, sizeFrame};
    root = new NodeBH(boundariesForRoot);
}

void BarnesHutStep::insertNode(NodeBH* node, NodeBH* quad) {
    if(!quad->isPoint() && !quad->wasInitialized()) {
      // jesli pusty node to dodajemy
      quad->setAttributes(node->getMass(), node->getIndex(), node->getX(), node->getY(), node->getZ());
      return;
    }
    if(quad->isPoint() && !quad->wasInitialized()) {
      // jesli external to pasowaloby wyodrebnic nowy i stary node do dwoch nowych branchy
        quad->setPoint(false);
        quad->pushPointFromQuadLower();
        std::array<double, 3> pos({node->getX(), node->getY(), node->getZ()});
        quad->updateCenterOfMass(node->getMass(), pos);

        int index = quad->numberOfSubCube(pos[0], pos[1], pos[2]);
        if(index < 8) 
        {
            insertNode(node, quad->getChild(index));
        }
        return;
    }
    if(!quad->isPoint() && quad->wasInitialized()) {
      // jesli jest internal to updejt masy i rekurencyjnie dalej
        std::array<double, 3> pos({node->getX(), node->getY(), node->getZ()});
        quad->updateCenterOfMass(node->getMass(), pos);
        
        int index = quad->numberOfSubCube(pos[0], pos[1], pos[2]);
        if(index<8)
        {
            insertNode(node, quad->getChild(index));
        }
    }
}

void BarnesHutStep::createTree(tf3& positions) {
    root->addQuads(root->getBoundaries());

    for(unsigned i=0; i<numberOfBodies; i++)
    {
        std::array<double, 3> pos;
        for(int jj=0; jj<3; jj++) pos[jj] = positions[i*3 + jj];
        root->updateCenterOfMass(weights[i], pos);

        int index = root->numberOfSubCube(pos[0], pos[1], pos[2]);
        if(index < 8) 
        {
            std::array<double, 6> bound = root->getChild(index)->getBoundaries();
            NodeBH* node = new NodeBH(weights[i], i, pos, root->getChild(index)->getBoundaries());
            insertNode(node, root->getChild(index));
            delete node;
        }
    }
}

void BarnesHutStep::DFS_BH(NodeBH* r, std::string indent) {
    r->setIndent(indent);
    indent = indent + "★";
    if(r->wasInitialized()) {
        for(auto* child : r->getQuads()) {
            DFS_BH(child, indent);
        }
    }
}

void BarnesHutStep::cleanUpTreePostOrder(NodeBH* r) {
    if(r->wasInitialized()) {
        for(auto* child : r->getQuads()) {
            cleanUpTreePostOrder(child);
        }       
    }
    delete r;
}

double BarnesHutStep::dist(double one, double two) {
    return fabs(two - one);
}

double BarnesHutStep::distanceBetweenTwoNodes(double x1, double y1, double z1, double x2, double y2, double z2) {
    double x = x2 - x1;
    double y = y2 - y1;
    double z = z2 - z1;
    return sqrt(x*x + y*y + z*z);
}

void BarnesHutStep::testingMomemntum() {
    float momentumX = 0.0f, momentumY = 0.0f, momentumZ = 0.0f;
    for (unsigned i = 0; i < numberOfBodies; i++) {
        momentumX += (weights[i] * velocities[i * 3]);
        momentumY += (weights[i] * velocities[i * 3 + 1]);
        momentumZ += (weights[i] * velocities[i * 3 + 2]);
    }
    //std::cout << "ZZP => " << momentumX << " " << momentumY << " " << momentumZ << std::endl;
} 

void BarnesHutStep::computeForceForBody(NodeBH* r, std::array<double, 3>& pos, int i)
{
    if(r->isPoint() && !r->wasInitialized())
    {
        if(r->getIndex() == i) return; // ten sam Node

        // Jesli node jest zewnetrzny, to policz sile ktora wywiera ten node na obecnie rozwazane cialo
        double distX = r->getSelectedPosition(0) - pos[0];
        double distY = r->getSelectedPosition(1) - pos[1];
        double distZ = r->getSelectedPosition(2) - pos[2];
        double dist = (distX * distX + distY * distY + distZ * distZ) + EPS * EPS;
        dist = dist * sqrt(dist);
        // jak sprawdzic czy to te same bodies? Musze zachowac i != j
        float F = G * (r->getMass() * weights[i]);
        forces[i * 3] += F * distX / dist; // force = G(m1*m2)/r^2
        forces[i * 3 + 1] += F * distY / dist;
        forces[i * 3 + 2] += F * distZ / dist;
    }
    else if(!r->isPoint() && r->wasInitialized())
    {
        if(r->isInQuad(pos[0], pos[1], pos[2])) 
        {
            //rekurencja pomijajac ratio
            for(auto* child : r->getQuads()) 
            {
                computeForceForBody(child, pos, i);
            }
            return;
        }

        std::array<double, 6>& boundaries = r->getBoundaries();
        
        double d = distanceBetweenTwoNodes(pos[0], pos[1], pos[2], 
            r->getSelectedCenterOfMass(0),
            r->getSelectedCenterOfMass(0),
            r->getSelectedCenterOfMass(0));
        double s = boundaries[1] - boundaries[0];

        bool isFarAway = (s/d < theta) ? true : false;
        /*
        bool isFarAway = false;
        for(int j = 0; j < 3; j++)
        {
            double d = dist(pos[j], r->getSelectedCenterOfMass(j));
            double s = boundaries[2*j + 1] - boundaries[2*j];
            if(BarnesHutStep::minS > s) BarnesHutStep::minS = s;
            if(s/d < theta) 
            {
                isFarAway = true;
            }
            else 
            {
                isFarAway = false;
                break;
            }
        }*/
        if(isFarAway)
        {
            double distX = r->getSelectedCenterOfMass(0) - pos[0];
            double distY = r->getSelectedCenterOfMass(1) - pos[1];
            double distZ = r->getSelectedCenterOfMass(2)- pos[2];
            double dist = (distX * distX + distY * distY + distZ * distZ) + EPS * EPS;
            dist = dist * sqrt(dist);
            // jak sprawdzic czy to te same bodies? Musze zachowac i != j
            float F = G * (r->getTotalMass() * weights[i]);
            forces[i * 3] += F * distX / dist; // force = G(m1*m2)/r^2
            forces[i * 3 + 1] += F * distY / dist;
            forces[i * 3 + 2] += F * distZ / dist;
        }
        else
        {
            for(auto* child : r->getQuads())
            {
                computeForceForBody(child, pos, i);
            }
        }
    }
}

void BarnesHutStep::compute(tf3 &positions, float dt)
{
    initializingRoot();
    createTree(positions);
    //std::string indent = "★";
    //DFS_BH(root, indent);
    std::fill(forces.begin(), forces.end(), 0.0);
        
    for(unsigned i=0; i<numberOfBodies; i++)
    {
        std::array<double, 3> arr({positions[i*3], positions[i*3 + 1], positions[i*3 + 2]});
        computeForceForBody(root, arr, i);
    }
    testingMomemntum();
        
    for (unsigned i = 0; i < numberOfBodies; i++) {
        for (int j = 0; j < 3; j++) {
                float acceleration = forces[i * 3 + j] / weights[i];
                positions[i * 3 + j] +=
                    velocities[i * 3 + j] * dt + acceleration * dt * dt / 2;
                velocities[i * 3 + j] += acceleration * dt;
        }
    }
    cleanUpTreePostOrder(root);
}
