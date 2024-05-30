#include <pbd_data/bou_tag.h>
#include <pbd_data/datastruct.h>
#include <pbd_data/edge_set.h>
#include <pbd_data/output_mesh_data.h>
#include <pbd_data/read_tet.h>
#include <pbd_data/test.h>
#include <pbd_data/tetEdgeIds.h>
#include <pbd_data/tetSurfaceTriIds.h>
#include <pbd_data/tet_set.h>

#include <iostream>
using namespace std;

void readinp_tet_ex4_1(string Path, vector<vert> &nodes, vector<tet> &tets) {
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
  cout << "read node" << endl;
  while (fin >> str) {
    if (str == "*NODE") break;
  }
  while (1) {
    while (1) {
      ch = fin.get();
      ch = fin.peek();
      if (ch == '*' || (ch <= '9' && ch >= '0')) break;
    }
    if (ch == '*') break;
    fin >> a >> c >> b1 >> c >> b2 >> c >> b3;
    tmp.x = b1;
    tmp.y = b2;
    tmp.z = b3;
    nodes.push_back(tmp);
  }
  cout << "read tet" << endl;
  while (fin >> str) {
    if (str.substr(0, 32) == "*ELEMENT,TYPE=C3D4,ELSET=auto001") break;
  }
  int meshid;
  tet tmp2{0, 0, 0, 0};
  while (1) {
    while (1) {
      ch = fin.get();
      ch = fin.peek();
      if (ch == '*' || (ch <= '9' && ch >= '0')) break;
    }
    if (ch == '*') break;
    fin >> meshid >> c;
    fin >> tmp2.v[0] >> c;
    fin >> tmp2.v[1] >> c;
    fin >> tmp2.v[2] >> c;
    fin >> tmp2.v[3];
    for (int i = 0; i < 4; i++) {
      tmp2.v[i]--;
    }
    tets.push_back(tmp2);
  }
  cout << "finish read" << endl;
  fin.close();
}

void out_node(string path, vector<vert> &verts) {
  ofstream fout;
  fout.open(path);

  int nv = verts.size();
  fout << nv << " " << 3 << " " << 0 << " " << 0 << endl;
  for (int i = 0; i < nv; i++) {
    fout << i << " ";
    fout << verts[i].x << " ";
    fout << verts[i].y << " ";
    fout << verts[i].z << endl;
  }

  fout.close();
}

void out_ele(string path, vector<tet> tets) {
  ofstream fout;
  fout.open(path);

  int ne = tets.size();
  fout << ne << " " << 4 << " " << 0 << endl;
  for (int i = 0; i < ne; i++) {
    fout << i << " ";
    fout << tets[i].v[0] << " ";
    fout << tets[i].v[1] << " ";
    fout << tets[i].v[2] << " ";
    fout << tets[i].v[3] << endl;
  }

  fout.close();
}

int main() {
  cout << "begin" << endl;
  string file_name = "40000elem_lv_rv";
  string file_format = ".inp";
  string input_path = "./data/input/" + file_name + file_format;
  string output_path_node = "./data/output/" + file_name + ".node";
  string output_path_ele = "./data/output/" + file_name + ".ele";

  vector<vert> verts;
  vector<tet> tets;
  // get verts and tetIds
  readinp_tet_ex4_1(input_path, verts, tets);

  out_node(output_path_node, verts);
  out_ele(output_path_ele, tets);

  return 0;
}
