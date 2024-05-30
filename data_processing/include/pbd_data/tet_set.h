#include <pbd_data/datastruct.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>
using namespace std;

struct get_tet_set {
  struct Tet {
    int a, b, c, d;
    int set_id;
  };
  vector<Tet> tet_sets;
  vector<bool> vis;
  int NV, NT, tmp;
  const vector<vert> &verts;
  const vector<tet> &tets;
  get_tet_set(const vector<vert> &verts, const vector<tet> &tets)
      : verts{verts}, tets{tets} {
    NV = verts.size();
    NT = tets.size();
    for (size_t i = 0; i < NT; i++) {
      Tet tmp;
      tmp.a = tets[i].v[0];
      tmp.b = tets[i].v[1];
      tmp.c = tets[i].v[2];
      tmp.d = tets[i].v[3];
      tmp.set_id = 0;
      tet_sets.push_back(tmp);
    }
    for (size_t i = 0; i < NV; i++) {
      vis.push_back(false);
    }
  }

  void tet_set_id(int &num_tet_set, vector<int> &tet_set_id) {
    int cnt = 1;
    while (1) {
      bool flag = false;
      // memset(vis,false,sizeof(vis));
      fill(vis.begin(), vis.end(), false);
      for (size_t i = 0; i < NT; i++) {
        if (tet_sets[i].set_id == 0) {
          int id1 = tet_sets[i].a;
          int id2 = tet_sets[i].b;
          int id3 = tet_sets[i].c;
          int id4 = tet_sets[i].d;

          if (!vis[id1] && !vis[id2] && !vis[id3] && !vis[id4]) {
            tet_sets[i].set_id = cnt;
            vis[id1] = true;
            vis[id2] = true;
            vis[id3] = true;
            vis[id4] = true;
          }
          flag = true;
        }
      }
      if (!flag) break;
      cnt++;
    }
    num_tet_set = cnt - 1;
    for (int i = 0; i < NT; i++) {
      tet_set_id.push_back(tet_sets[i].set_id);
    }

    vector<Tet>().swap(tet_sets);
    vector<bool>().swap(vis);
  }
};
