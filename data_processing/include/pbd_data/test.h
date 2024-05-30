#pragma once
#include<iostream>
#include<pbd_data/datastruct.h>
using namespace std;

void test_vert(vector<vert> &nodes)
{
    for (int i = 0; i < nodes.size(); i++)
    // for (int i = 0; i < 1; i++)
    {
        cout << nodes[i].x << " " << nodes[i].y << " " << nodes[i].z << endl;
    }
}

void test_tet(vector<tet> &tets)
{
    for (int i = 0; i < tets.size(); i++)
    // for (int i = 0; i < 1; i++)
    {
        cout << tets[i].v[0] << " " << tets[i].v[1] << " " << tets[i].v[2] << " " << tets[i].v[3] << endl;
    }
}