#include "NodeBH.h"
#include <cstdio>

/* Constructor to creating complete node, which is ready to be quadrant. */
NodeBH::NodeBH(double mass, std::vector<double> pos, std::vector<double> boards) :
      mass(mass), hasPoint(true)
{
    initializeVectors();
    this->pos = pos;
    this->boards = boards;
    totalMass += mass;
    for(int i=0; i<3; i++)
    {
        this->pos[i] = pos[i];
        this->boards[2*i] = boards[2*i];
        this->boards[2*i + 1] = boards[2*i + 1];
        numerator[i] += mass*pos[i];
        centerOfMass[i] = numerator[i]/totalMass;
    }
}

/* Constructor to creating empty node, which is ready to be quadrant. */
NodeBH::NodeBH(double* board) {
    initializeVectors();
    for(int i=0; i<6; i++) {
      boards[i] = board[i];
    }
    mass = 0;
    totalMass = 0;
    hasPoint = false;
}

NodeBH::~NodeBH() {
    quads.clear();
    centerOfMass.clear();
    numerator.clear();
    boards.clear();
    pos.clear();
    //bodies.clear();
}

void NodeBH::initializeVectors() {
    pos.resize(3);
    boards.resize(6);
    numerator.resize(3);
    centerOfMass.resize(3);
}

void NodeBH::addQuads(std::vector<double> b)
{
    double boardsForQuad[6] = {b[x1], b[x1] + (b[x2]-b[x1])/2, b[y1], b[y1] + (b[y2]-b[y1])/2, b[z1], b[z1] + (b[z2] - b[z1])/2};
    quads.push_back(new NodeBH(boardsForQuad));

    double boardsForQuad2[6] = {b[x1] + (b[x2]-b[x1])/2, b[x2], b[y1], b[y1] + (b[y2]-b[y1])/2, b[z1], b[z1] + (b[z2] - b[z1])/2};
    quads.push_back(new NodeBH(boardsForQuad2));

    double boardsForQuad3[6] = {b[x1], b[x1] + (b[x2]-b[x1])/2, b[y1] + (b[y2]-b[y1])/2, b[y2], b[z1], b[z1] + (b[z2] - b[z1])/2};
    quads.push_back(new NodeBH(boardsForQuad3));

    double boardsForQuad4[6] = {b[x1] + (b[x2]-b[x1])/2, b[x2], b[y1] + (b[y2]-b[y1])/2, b[y2], b[z1], b[z1] + (b[z2] - b[z1])/2};
    quads.push_back(new NodeBH(boardsForQuad4));

    double boardsForQuad5[6] = {b[x1], b[x1] + (b[x2]-b[x1])/2, b[y1], b[y1] + (b[y2]-b[y1])/2, b[z1] + (b[z2] - b[z1])/2, b[z2]};
    quads.push_back(new NodeBH(boardsForQuad5));

    double boardsForQuad6[6] = {b[x1] + (b[x2]-b[x1])/2, b[x2], b[y1], b[y1] + (b[y2]-b[y1])/2, b[z1] + (b[z2] - b[z1])/2, b[z2]};
    quads.push_back(new NodeBH(boardsForQuad6));

    double boardsForQuad7[6] = {b[x1], b[x1] + (b[x2]-b[x1])/2, b[y1] + (b[y2]-b[y1])/2, b[y2], b[z1] + (b[z2] - b[z1])/2, b[z2]};
    quads.push_back(new NodeBH(boardsForQuad7));

    double boardsForQuad8[6] = {b[x1] + (b[x2]-b[x1])/2, b[x2], b[y1] + (b[y2]-b[y1])/2, b[y2], b[z1] + (b[z2] - b[z1])/2, b[z2]};
    quads.push_back(new NodeBH(boardsForQuad8));
}

int NodeBH::getNumberOfQuads() {
    return quads.size();
}

std::vector<NodeBH*> NodeBH::getQuads() {
    return quads;
}

bool NodeBH::isInQuad(double x, double y, double z) {
    if(!(x>=boards[0] && x<=boards[1])) return false;
    if(!(y>=boards[2] && y<=boards[3])) return false;
    if(!(z>=boards[4] && z<=boards[5])) return false;
    return true;
}

bool NodeBH::isPoint() {
    return hasPoint;
}

void NodeBH::setAttributes(double mass, double x, double y, double z) {
    hasPoint = true;
    this->mass = mass;
    pos[0] = x;
    pos[1] = y;
    pos[2] = z;
    totalMass += mass;
    for(int i=0; i<3; i++)
    {
        numerator[i] += mass*pos[i];
        centerOfMass[i] = numerator[i]/totalMass;
    }
}

void NodeBH::updateCenterOfMass(double mass, std::vector<double> pos) {
    totalMass += mass;
    for(int i=0; i<3; i++)
    {
        numerator[i] += mass*pos[i];
        centerOfMass[i] = numerator[i]/totalMass;
    }
}

void NodeBH::pushQuadsLower() {
    addQuads(boards);
    for(auto* child : quads) {
        if(child->isInQuad(pos[0], pos[1], pos[2])) {
            child->setAttributes(mass, pos[0], pos[1], pos[2]); // TUTAJ MOGLEM ZBUGOWAC
        }
    }
}

std::vector<double> NodeBH::getBoards() {
    return boards;
}

double NodeBH::getX() {
    return pos[0];
}

double NodeBH::getY() {
    return pos[1];
}

double NodeBH::getZ() {
    return pos[2];
}

double NodeBH::getMass() {
    return mass;
}

double NodeBH::getTotalMass() {
    return totalMass;
}

double NodeBH::getSelectedCenterOfMass(int i) {
    return centerOfMass[i];
}

double NodeBH::getSelectedPosition(int i) {
    return pos[i];
}

void NodeBH::setPoint(bool val) {
    hasPoint = val;
}
