#ifndef NODEBH_H
#define NODEBH_H

#include <array>
#include <vector>
#include <string>


class NodeBH {
public:
    NodeBH() = delete;
    NodeBH(float mass, long long id, std::array<float, 3>& pos, std::array<float, 6>& boundaries);
    NodeBH(std::array<float, 6>& board) {
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
    std::array<float, 6>& getBoundaries() {
        return boundaries;
    }
    bool wasInitialized() {
      return childrenExists;
    }
    int getNumberOfQuads() {
        return quads.size();
    }
    float getX() {
        return pos[0];
    }
    float getY() {
        return pos[1];
    }
    float getZ() {
        return pos[2];
    }
    float getMass() {
        return mass;
    }
    float getTotalMass() {
        return totalMass;
    }
    float getSelectedCenterOfMass(int i) {
        return centerOfMass[i];
    }
    float getSelectedPosition(int i) {
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
    bool isInQuad(float x, float y, float z);
    void addQuads(std::array<float, 6>& b);
    int numberOfSubCube(float x, float y, float z);
    void pushPointFromQuadLower();
    void setAttributes(float mass, long long id, float x, float y, float z);
    void updateCenterOfMass(float mass, std::array<float, 3>& pos);

private:
    float mass;
    float totalMass;
    bool hasPoint;
    bool childrenExists;
    std::string indent;
    long long id;
    std::array<float, 6> boundaries;
    std::array<float, 3> pos;
    std::array<float, 3> centerOfMass;
    std::array<float, 3> numerator;
    std::array<NodeBH*, 8> quads;
};

#endif
