#include "BarnesHutStep.h"

BarnesHutStep::BarnesHutStep(std::vector<float>& masses, unsigned numberOfBodies) {
    this->numberOfBodies = numberOfBodies;
    weights.resize(numberOfBodies);
    forces.resize(3*numberOfBodies);

    double boardsForRoot[6] = {-sizeFrame, sizeFrame, -sizeFrame, sizeFrame, -sizeFrame, sizeFrame};
    root = new NodeBH(boardsForRoot);

    weights = std::vector<double>(masses.begin(), masses.end());

    //randomGenerator = new RandomGenerators();
    //randomGenerator->initializeVelocities<std::vector<float>>(velocities, numberOfBodies);
}

BarnesHutStep::~BarnesHutStep() {

}

bool BarnesHutStep::isCreate() const{
    return wasCreated;
}

void BarnesHutStep::setCreate() {
    wasCreated = true;
}

void BarnesHutStep::insertNode(NodeBH* node, NodeBH* quad) {
    if(!quad->getNumberOfQuads() && !quad->getTotalMass()) {
      // jesli pusty node to dodajemy
      quad->setAttributes(node->getMass(), node->getX(), node->getY(), node->getZ());
      return;
    }

    if(!quad->getNumberOfQuads() && quad->getTotalMass()) {
      // jesli external to pasowaloby wyodrebnic nowy i stary node do dwoch nowych branchy
      quad->setPoint(false);
      quad->pushQuadsLower(); // JESZCZE TO UZUPELNIC
      std::vector<double> pos({node->getX(), node->getY(), node->getZ()});
      quad->updateCenterOfMass(node->getMass(), pos);
      for(auto* child : quad->getQuads()) {
          if(child->isInQuad(pos[0], pos[1], pos[2])) {
              insertNode(node, child);
              break;
          }
      }
      return;
    }

    if(quad->getNumberOfQuads()) {
      // jesli jest internal to updejt masy i rekurencyjnie dalej
      // poprawienie mass
        std::vector<double> pos({node->getX(), node->getY(), node->getZ()});
        quad->updateCenterOfMass(node->getMass(), pos);
        for(auto* child : quad->getQuads()) {
            if(child->isInQuad(pos[0], pos[1], pos[2])) {
                insertNode(node, child);
                break;
            }
        }
    }
}

void BarnesHutStep::createTree(tf3& positions) {
    setCreate();
    root->addQuads(root->getBoards());
    std::vector<double> boards(6);
    for(unsigned i=0; i<numberOfBodies; i++)
    {
        std::vector<double> pos(3);
        for(int jj=0; jj<3; jj++) pos[jj] = positions[i*3 + jj];

        // check if point is in the biggest quadrant
        root->updateCenterOfMass(weights[i], pos); // problems for big N

        for(auto* child : root->getQuads()) {
            if(child->isInQuad(pos[0], pos[1], pos[2])) {
                boards = child->getBoards();
                NodeBH* node = new NodeBH(weights[i], pos, boards);
                insertNode(node, child);
                break;
            }
        }
    }
}

void BarnesHutStep::DFS_BH(NodeBH* r) {
    if(r->getTotalMass() > 0)
    std::cout << r->getSelectedCenterOfMass(0) << " " << r->getSelectedCenterOfMass(1) << " " << r->getSelectedCenterOfMass(2) << std::endl;
    //std::cout << r->getNumberOfQuads() << " " << r->isPoint() << " " << r->getTotalMass() << std::endl;
    for(auto* child : r->getQuads()) {
        DFS_BH(child);
    }
    //std::cout << "\n" << "\n";
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

/*
To calculate the net force acting on body b, use the following recursive procedure, starting with the root of the quad-tree:
If the current node is an external node (and it is not body b), calculate the force exerted by the current node on b,
and add this amount to b’s net force.
Otherwise, calculate the ratio s/d. If s/d < θ, treat this internal node as a single body,
and calculate the force it exerts on body b, and add this amount to b’s net force.
Otherwise, run the procedure recursively on each of the current node’s children.
*/

void BarnesHutStep::computeForceForBody(NodeBH* r, std::vector<double> pos, int i)
{
    if(!r->getNumberOfQuads() && r->getTotalMass())
    {
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
    else
    {
        std::vector<double> boards = r->getBoards();
        bool isFarAway = false;
        for(int j = 0; j < 3; j++)
        {
            double d = dist(pos[j], r->getSelectedCenterOfMass(j));
            double s = boards[2*j + 1] - boards[2*j];
            // std::cout << s/d << std::endl;
            if(s/d < theta) isFarAway = true;
        }
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
    if(!isCreate())
    {
        createTree(positions); // tylko raz i chyba lepiej w bridge'u
        // DFS_BH(root);
        std::fill(forces.begin(), forces.end(), 0.0);
        for(unsigned i=0; i<numberOfBodies; i++)
        {
            std::vector<double> vec({positions[i*3], positions[i*3 + 1], positions[i*3 + 2]});
            computeForceForBody(root, vec, i);
        }
        for(unsigned i=0; i<numberOfBodies; i++)
        {
            if(i<5)
            {
                for(int j=0; j<3; j++)
                {
                    std::cout << forces[i*3+j] << " ";
                }
                std::cout << std::endl;
            }
        }
    }
    //copy(positions, positionsAfterStep);
}
