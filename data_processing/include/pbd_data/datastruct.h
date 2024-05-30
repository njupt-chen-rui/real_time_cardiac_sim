#pragma once
#include<iostream>
#include<vector>
#include<array>
#include<memory.h>
#include<math.h>
#include<algorithm>

struct vert{
    double x, y, z;
};

struct tri{
    int v[3];
};

struct vec3{
    double x, y, z;
    vec3(){
        x = y = z = 0.0;
    }
    vec3(double x, double y, double z): x{x}, y{y}, z{z} {}
};

struct surf_tri{
    int v[3];
};

struct tet{
    int v[4];
    int set_id;
    bool operator<(const tet&other)const{
        return set_id < other.set_id;
    }
};