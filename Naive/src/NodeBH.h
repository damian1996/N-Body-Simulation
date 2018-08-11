#ifndef NODEBH_H
#define NODEBH_H

#include <array>
#include <vector>
#include <string>

class NodeBH {
public:
    NodeBH() = delete;
    NodeBH(double mass, std::array<double, 3>& pos, std::array<double, 6>& boundaries);
    NodeBH(std::array<double, 6>& boundaries);
    ~NodeBH();
    void addQuads(std::array<double, 6>& b);
    bool isInQuad(double x, double y, double z);
    bool isPoint();
    void pushPointFromQuadLower();
    void setAttributes(double mass, double x, double y, double z);
    void updateCenterOfMass(double mass, std::array<double, 3>& pos);

    std::array<double, 6>& getBoundaries();
    std::array<NodeBH*, 8>& getQuads();
    bool wasInitialized();
    int getNumberOfQuads();
    double getX();
    double getY();
    double getZ();
    double getMass();
    double getTotalMass();
    double getSelectedCenterOfMass(int i);
    double getSelectedPosition(int i);
    void setPoint(bool val);
    std::string getIndent();
    void setIndent(std::string str);

private:
    double mass;
    double totalMass;
    bool hasPoint;
    bool childrenExists;
    std::string indent;
    int id;
    std::array<double, 6> boundaries;
    std::array<double, 3> pos;
    std::array<double, 3> centerOfMass;
    std::array<double, 3> numerator;
    std::array<NodeBH*, 8> quads;
};

#endif
