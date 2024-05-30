#pragma once
#include<iostream>
#include<vector>
#include<math.h>
#include<pbd_data/datastruct.h>
using namespace std;

void get_bou_tag_1(vector<vert> &verts, vector<int> &bou_tag){
    int NV = verts.size();
    double eps = 1e-3;
    int cnt = 0;
    for(size_t i = 0; i < NV; i++){
        if(fabs(verts[i].y - 5.0) < eps){
            bou_tag.push_back(1);
            cnt ++;
        }else{
            bou_tag.push_back(0);
        }
    }
    cout << "dirichlet bou base has " << cnt << "verts" << endl;
}

void get_bou_tag_2(vector<vert> &verts, vector<int> &bou_tag){
    int NV = verts.size();
    double eps = 1e-3;
    for(size_t i = 0; i < NV; i++){
        // if(fabs(verts[i].z) < eps){
        //     bou_tag.push_back(1);
        // }else{
        //     bou_tag.push_back(0);
        // }
        bou_tag.push_back(0);
    }
}