#include "NodeBH.h"
#include <cstdio>

NodeBH::NodeBH(float mass, long long id, std::array<float, 3>& pos, std::array<float, 6>& boundaries) :
      mass(mass), totalMass(0), hasPoint(true), childrenExists(false), id(id)
{
    std::copy(std::begin(pos), std::end(pos), std::begin(this->pos));
    std::copy(std::begin(boundaries), std::end(boundaries), std::begin(this->boundaries));
    totalMass = mass;
    for(int i=0; i<3; i++)
    {
        this->pos[i] = pos[i];
        this->boundaries[2*i] = boundaries[2*i];
        this->boundaries[2*i + 1] = boundaries[2*i + 1];
        numerator[i] = mass*pos[i];
        centerOfMass[i] = numerator[i]/totalMass;
    }
}

bool NodeBH::isInQuad(float x, float y, float z) {
    if(!(x>=boundaries[0] && x<=boundaries[1])) return false;
    if(!(y>=boundaries[2] && y<=boundaries[3])) return false;
    if(!(z>=boundaries[4] && z<=boundaries[5])) return false;
    return true;
}

void NodeBH::addQuads(std::array<float, 6>& b)
{
    childrenExists = true;
    std::array<float, 6> boundariesForQuad = {b[0], b[0] + (b[1]-b[0])/2, b[2], b[2] + (b[3]-b[2])/2, b[4], b[4] + (b[5] - b[4])/2};
    quads[0] = new NodeBH(boundariesForQuad);

    boundariesForQuad = {b[0] + (b[1]-b[0])/2, b[1], b[2], b[2] + (b[3]-b[2])/2, b[4], b[4] + (b[5] - b[4])/2};
    quads[1] = new NodeBH(boundariesForQuad);

    boundariesForQuad = {b[0], b[0] + (b[1]-b[0])/2, b[2] + (b[3]-b[2])/2, b[3], b[4], b[4] + (b[5] - b[4])/2};
    quads[2] = new NodeBH(boundariesForQuad);

    boundariesForQuad = {b[0] + (b[1]-b[0])/2, b[1], b[2] + (b[3]-b[2])/2, b[3], b[4], b[4] + (b[5] - b[4])/2};
    quads[3] = new NodeBH(boundariesForQuad);

    boundariesForQuad = {b[0], b[0] + (b[1]-b[0])/2, b[2], b[2] + (b[3]-b[2])/2, b[4] + (b[5] - b[4])/2, b[5]};
    quads[4] = new NodeBH(boundariesForQuad);

    boundariesForQuad = {b[0] + (b[1]-b[0])/2, b[1], b[2], b[2] + (b[3]-b[2])/2, b[4] + (b[5] - b[4])/2, b[5]};
    quads[5] = new NodeBH(boundariesForQuad);

    boundariesForQuad = {b[0], b[0] + (b[1]-b[0])/2, b[2] + (b[3]-b[2])/2, b[3], b[4] + (b[5] - b[4])/2, b[5]};
    quads[6] = new NodeBH(boundariesForQuad);

    boundariesForQuad = {b[0] + (b[1]-b[0])/2, b[1], b[2] + (b[3]-b[2])/2, b[3], b[4] + (b[5] - b[4])/2, b[5]};
    quads[7] = new NodeBH(boundariesForQuad);
}

int NodeBH::numberOfSubCube(float x, float y, float z) {
    int result = 0;
    if(x<boundaries[0] || x>boundaries[1]) return 8;
    if(y<boundaries[2] || y>boundaries[3]) return 8;
    if(z<boundaries[4] || z>boundaries[5]) return 8;

    if(z >= (boundaries[4] + (boundaries[5] - boundaries[4])/2)) result += 4;
    if(y >= (boundaries[2] + (boundaries[3] - boundaries[2])/2)) result += 2;
    if(x >= (boundaries[0] + (boundaries[1] - boundaries[0])/2)) result += 1;
    return result;
}

void NodeBH::setAttributes(float mass, long long id, float x, float y, float z) {
    hasPoint = true;
    this->mass = mass;
    this->id = id;
    pos[0] = x;
    pos[1] = y;
    pos[2] = z;
    totalMass = mass;
    for(int i=0; i<3; i++)
    {
        numerator[i] = mass*pos[i];
        centerOfMass[i] = numerator[i]/totalMass;
    }
}

void NodeBH::updateCenterOfMass(float mass, std::array<float, 3>& pos) {
    totalMass += mass;
    for(int i=0; i<3; i++)
    {
        numerator[i] += mass*pos[i];
        centerOfMass[i] = numerator[i]/totalMass;
    }
}

void NodeBH::pushPointFromQuadLower() {
    addQuads(boundaries);
    for(auto* child : quads) {
        if(child->isInQuad(pos[0], pos[1], pos[2])) {
            child->setAttributes(mass, id, pos[0], pos[1], pos[2]);
        }
    }
    mass = 0;
}
