#ifndef NODEBH_H
#define NODEBH_H

#include <array>
#include <vector>
#include <string>


class NodeBH {
public:
    NodeBH() = delete;
    NodeBH(double mass, long long id, std::array<double, 3>& pos, std::array<double, 6>& boundaries);
    NodeBH(std::array<double, 6>& board) {
        for(int i=0; i<6; i++) {
          boundaries[i] = board[i];
        }
        std::fill(numerator.begin(), numerator.end(), 0.0);
        mass = 0;
        totalMass = 0;
        hasPoint = false;
        childrenExists = false;
    }
    ~NodeBH() {

    }
    std::array<double, 6>& getBoundaries() {
        return boundaries;
    }
    bool wasInitialized() {
      return childrenExists;
    }
    int getNumberOfQuads() {
        return quads.size();
    }
    double getX() {
        return pos[0];
    }
    double getY() {
        return pos[1];
    }
    double getZ() {
        return pos[2];
    }
    double getMass() {
        return mass;
    }
    double getTotalMass() {
        return totalMass;
    }
    double getSelectedCenterOfMass(int i) {
        return centerOfMass[i];
    }
    double getSelectedPosition(int i) {
        return pos[i];
    }
    void setPoint(bool val) {
        hasPoint = val;
    }
    std::string getIndent() {
        return indent;
    }
    void setIndent(std::string str) {
        indent = str;
    }
    long long getIndex() {
        return id;
    }
    NodeBH* getChild(int i) {
        return quads[i];
    }
    bool isPoint() {
      return hasPoint;
    }
    std::array<NodeBH*, 8>& getQuads() {
        return quads;
    }
    bool isInQuad(double x, double y, double z);
    void addQuads(std::array<double, 6>& b);
    int numberOfSubCube(double x, double y, double z);
    void pushPointFromQuadLower();
    void setAttributes(double mass, long long id, double x, double y, double z);
    void updateCenterOfMass(double mass, std::array<double, 3>& pos);

private:
    double mass;
    double totalMass;
    bool hasPoint;
    bool childrenExists;
    std::string indent;
    long long id;
    std::array<double, 6> boundaries;
    std::array<double, 3> pos;
    std::array<double, 3> centerOfMass;
    std::array<double, 3> numerator;
    std::array<NodeBH*, 8> quads;
};

#endif
