#ifndef BODY_H
#define BODY_H

struct Body {
    double x, y, z, mass;
    Body(double x, double y, double z, double mass);
    ~Body();
};

#endif
