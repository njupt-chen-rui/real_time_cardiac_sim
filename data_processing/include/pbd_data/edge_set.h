#pragma once
#include<iostream>
#include<cstdio>
#include<cstring>
#include<pbd_data/datastruct.h>
using namespace std;

struct get_edge_set{
    struct Edge{
        int a,b;
        int set_id;
    };
    vector<Edge> edge;
    const vector<vert> &verts;
    const vector<int> &edge_inds;
    int NV, NE;
    vector<bool> vis;
    
    get_edge_set(const vector<vert> &verts, const vector<int> &edges) : verts{verts}, edge_inds{edges} {
        NV = verts.size();
        NE = edge_inds.size() / 2;
        for(int i = 0; i < NE; i++){
            Edge tmp;
            tmp.a = edge_inds[i * 2];
            tmp.b = edge_inds[i * 2 + 1];
            tmp.set_id = 0;
            edge.push_back(tmp);
        }
        for(int i = 0; i < NV; i++){
            vis.push_back(false);
        }
        // for(int i=0;i<10;i++){
        //     cout<<edge[i].a <<" "<<edge[i].b<<endl;
        // }
        // cout<<"NE1:"<<NE<<endl;
    }

    void edge_set(int &num_edge_set, vector<int> &edge_set_id){
        // cout<<"NE2:"<<NE<<endl;
        int cnt = 1;
        while(1){
            bool flag = false;
            // memset(vis, false, sizeof(vis));
            fill(vis.begin(), vis.end(), false);
            for(int i = 0; i < NE; i++){
                if(edge[i].set_id==0){
                    int id1 = edge[i].a;
                    int id2 = edge[i].b;
                    if(!vis[id1]&&!vis[id2]){
                        edge[i].set_id = cnt;
                        vis[id1] = true;
                        vis[id2] = true;
                    }
                    flag = true;
                }
            }
            if(!flag) break;
            cnt++;
        }
        num_edge_set = cnt - 1;
        // cout<<"NE3:"<<NE<<endl;
        for(int i = 0; i < NE; i++){
            edge_set_id.push_back(edge[i].set_id);
        }
        // cout<< "finish edge_set"<< endl;
        vector<Edge>().swap(edge);
        vector<bool>().swap(vis);
    }
};
