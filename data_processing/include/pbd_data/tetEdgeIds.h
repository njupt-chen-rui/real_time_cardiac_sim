#include<iostream>
#include<vector>
#include<set>
#include<pbd_data/datastruct.h>
using namespace std;

void tetEdgeIds(vector<vert> &verts, vector<tet> &tets, vector<int> &res){
    vector<set<int>> ans;
    int NV = verts.size();
    for(size_t i = 0;i < NV;i++){
        set<int> tmp;
        ans.push_back(tmp);
    }
    int n = tets.size();
    int a,b,c,d;
    for(int i=0;i<n;i++){
        for(int j=0;j<3;j++){
            for(int k=j+1;k<4;k++){
                int a = tets[i].v[j];
                int b = tets[i].v[k];
                if(a < b) ans[a].insert(b);
                else ans[b].insert(a);
            }
        }
    }
    for(int i=0;i<NV;i++){
        for(auto it = ans[i].cbegin();it!=ans[i].cend();it++){
            res.push_back(i);
            res.push_back(*it);
        }
    }
}