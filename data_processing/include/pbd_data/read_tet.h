#pragma once
#include<iostream>
#include<fstream>
#include<vector>
#include<array>
#include<memory.h>
#include<math.h>
#include<algorithm>
#include<pbd_data/datastruct.h>

using namespace std;

void readinp_tet(string Path, vector<vert> &nodes, vector<tet> &tets){
    ifstream fin;
    fin.open(Path);
    if (!fin.good()) {
        cout << "file open failed";
        exit(1);
    }
    string str, c;
    char ch;
    int a;
    double b1, b2, b3;
    vert tmp{0, 0, 0};
    cout<<"read node"<<endl;
    while(fin>>str){
        if(str=="*NODE") break;
    }
    while(1){
        while(1){
            ch = fin.get();
            ch = fin.peek();
            if(ch == '*' || (ch <= '9' && ch >= '0')) break;
        }
        if(ch == '*') break;
        fin>>a>>c>>b1>>c>>b2>>c>>b3;
        tmp.x = b1;
        tmp.y = b2;
        tmp.z = b3;
        nodes.push_back(tmp);
    }
    cout<<"read tet"<<endl;
    while(fin>>str){
        if(str.substr(0, 8)=="*ELEMENT") break;
    }
    int meshid;
    tet tmp2{0, 0, 0, 0};
    while(1){
        while(1){
            ch = fin.get();
            ch = fin.peek();
            if(ch == '*' || (ch <= '9' && ch >= '0')) break;
        }
        if(ch == '*') break;
        fin>>meshid>>c;
        fin>>tmp2.v[0]>>c;
        fin>>tmp2.v[1]>>c;
        fin>>tmp2.v[2]>>c;
        fin>>tmp2.v[3];
        for(int i=0;i<4;i++){
            tmp2.v[i]--;
        }
        tets.push_back(tmp2);
    }
    cout<<"finish read"<<endl;
    fin.close();
}

// TODO
void readinp_tri(){

}

void readpy_tet(string Path, vector<vert> &nodes, vector<tet> &tets, vector<vec3> &fiber, vector<vec3> &sheet, vector<vec3> &normal){
    ifstream fin;
    fin.open(Path);
    if (!fin.good()) {
        cout << "file open failed";
        exit(1);
    }

    string str, c;
    char ch;
    int a;
    double b1, b2, b3;
    vert tmp{0, 0, 0};
    cout<<"read node"<<endl;
    while(fin>>str){
        if(str=="'verts':") break;
    }
    fin>>str;
    while(1)
    {
        fin>>tmp.x>>c>>tmp.y>>c>>tmp.z>>c;
        nodes.push_back(tmp);
        // cout << b1 << " " << b2<<" "<<b3<<endl;
        if(c == "],") break;
    }

    cout<<"read tet"<<endl;
    fin>>str; // 'tetIds':
    fin>>str; // [
    int meshid;
    tet tmp2{0, 0, 0, 0};
    while(1)
    {
        fin >> tmp2.v[0] >> c >> tmp2.v[1] >> c >> tmp2.v[2] >> c >> tmp2.v[3] >> c;
        for(int i=0;i<4;i++){
            tmp2.v[i];
        }
        tets.push_back(tmp2);
        if(c == "],") break;
    }

    cout<<"read fiber"<<endl;
    while(fin>>str){
        if(str=="'fiberDirection':") break;
    }
    fin>>str;
    vec3 tmp3{0, 0, 0};
    while(1)
    {
        fin>>tmp3.x>>c>>tmp3.y>>c>>tmp3.z>>c;
        fiber.push_back(tmp3);
        if(c == "],") break;
    }

    cout<<"read sheet"<<endl;
    while(fin>>str){
        if(str=="'sheetDirection':") break;
    }
    fin>>str;
    while(1)
    {
        fin>>tmp3.x>>c>>tmp3.y>>c>>tmp3.z>>c;
        sheet.push_back(tmp3);
        if(c == "],") break;
    }

    cout<<"read normal"<<endl;
    while(fin>>str){
        if(str=="'normalDirection':") break;
    }
    fin>>str;
    while(1)
    {
        fin>>tmp3.x>>c>>tmp3.y>>c>>tmp3.z>>c;
        normal.push_back(tmp3);
        if(c == "],") break;
    }

    cout<<"finish read"<<endl;
    fin.close();
}