#ifndef NODEBH_H
#define NODEBH_H

#include <vector>
//#include "Body.h"

enum class Board { x1, x2, y1, y2, z1, z2 };

class NodeBH {
public:
    NodeBH() = delete;
    NodeBH(double mass, std::vector<double> pos, std::vector<double> boards);
    NodeBH(double* boards);
    ~NodeBH();

private:
    void initializeVectors();

public:
    void addQuads(std::vector<double> b);
    bool isInQuad(double x, double y, double z);
    bool isPoint();
    void pushQuadsLower();
    void setAttributes(double mass, double x, double y, double z);
    void updateCenterOfMass(double mass, std::vector<double> pos);

    std::vector<double> getBoards();
    std::vector<NodeBH*> getQuads();
    int getNumberOfQuads();
    double getX();
    double getY();
    double getZ();
    double getMass();
    double getTotalMass();
    double getSelectedCenterOfMass(int i);
    double getSelectedPosition(int i);
    void setPoint(bool val);

private:
    double mass;
    bool hasPoint;
    std::vector<double> boards, pos, centerOfMass, numerator;
    std::vector<NodeBH*> quads;
    double totalMass;
    const int numberOfQuads = 8;
    const int x1 = static_cast<const int>(Board::x1);
    const int x2 = static_cast<const int>(Board::x2);
    const int y1 = static_cast<const int>(Board::y1);
    const int y2 = static_cast<const int>(Board::y2);
    const int z1 = static_cast<const int>(Board::z1);
    const int z2 = static_cast<const int>(Board::z2);
    //std::vector<Body*> bodies;
};

#endif
