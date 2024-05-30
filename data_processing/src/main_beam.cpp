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

void readinp_tet_beam(string Path, vector<vert> &nodes, vector<tet> &tets) {
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
  cout << "begin read node" << endl;
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

  cout << "begin read tet" << endl;
  while (fin >> str) {
    if (str == "*ELEMENT,TYPE=C3D4,ELSET=auto1") break;
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

void read_fiber_from_txt_ex4_2(string path, int nv, vector<vec3> &fiber,
                               vector<vec3> &sheet, vector<vec3> &normal) {
  ifstream fin;
  fin.open(path);
  if (!fin.good()) {
    cout << "file open failed";
    exit(1);
  }

  int vert_id;
  for (int i = 0; i < nv; i++) {
    fin >> vert_id;
    fin >> fiber[i].x;
    fin >> fiber[i].y;
    fin >> fiber[i].z;
    fin >> sheet[i].x;
    fin >> sheet[i].y;
    fin >> sheet[i].z;
    fin >> normal[i].x;
    fin >> normal[i].y;
    fin >> normal[i].z;
  }

  fin.close();
}

vec3 vec3_cross(vec3 &a, vec3 &b) {
  vec3 res;
  res.x = a.y * b.z - a.z * b.y;
  res.y = a.z * b.x - a.x * b.z;
  res.z = a.x * b.y - a.y * b.x;
  return res;
}

int main() {
  cout << "begin" << endl;
  string file_name = "beam";
  string file_format = ".inp";
  string input_path = "./data/input/" + file_name + file_format;
  string output_path = "./data/output/" + file_name + ".py";

  output_mesh_data out(output_path);
  // output name
  out.output_name(file_name);
  cout << "finish out name" << endl;

  vector<vert> verts;
  vector<tet> tets;
  vector<int> bou_dirichlet;

  // get verts and tetIds
  readinp_tet_beam(input_path, verts, tets);

  int nv = verts.size();  // num of vertexes

  // output verts
  out.output_verts(verts);
  cout << "finish out verts" << endl;

  // get tet_set
  int num_tet_set;
  vector<int> tets_set_id;
  get_tet_set gts(verts, tets);
  gts.tet_set_id(num_tet_set, tets_set_id);
  for (int i = 0; i < tets.size(); i++) {
    tets[i].set_id = tets_set_id[i];
  }
  sort(tets.begin(), tets.end());
  sort(tets_set_id.begin(), tets_set_id.end());

  // output tetIds
  out.output_tetIds(tets);
  cout << "finish out tetIds" << endl;

  // // get tetEdgeIds
  // vector<int> tet_edge_ids;
  // tetEdgeIds(verts, tets, tet_edge_ids);

  // // output tetEdgeIds
  // out.output_tetEdgeIds(tet_edge_ids);
  // cout << "finish out tetEdgeIds" << endl;

  vector<vec3> fiber, sheet, normal;
  int ne = tets.size();
  fiber.resize(ne);
  sheet.resize(ne);
  normal.resize(ne);
  double eps = 1e-12;
  for (int i = 0; i < ne; i++) {
    fiber[i].x = 1.0;
    fiber[i].y = 0.0;
    fiber[i].z = 0.0;

    sheet[i].x = 0.0;
    sheet[i].y = 1.0;
    sheet[i].z = 0.0;

    double len_fiber = sqrt(fiber[i].x * fiber[i].x + fiber[i].y * fiber[i].y +
                            fiber[i].z * fiber[i].z);
    if (abs(len_fiber) < eps) {
      cout << "len_fiber failed" << endl;
      exit(1);
    }
    fiber[i].x /= len_fiber;
    fiber[i].y /= len_fiber;
    fiber[i].z /= len_fiber;

    double len_sheet = sqrt(sheet[i].x * sheet[i].x + sheet[i].y * sheet[i].y +
                            sheet[i].z * sheet[i].z);
    if (abs(len_sheet) < eps) {
      cout << "len_sheet failed" << endl;
      exit(1);
    }
    sheet[i].x /= len_sheet;
    sheet[i].y /= len_sheet;
    sheet[i].z /= len_sheet;

    vec3 tmp = vec3_cross(fiber[i], sheet[i]);
    double len_tmp = sqrt(tmp.x * tmp.x + tmp.y * tmp.y + tmp.z * tmp.z);
    tmp.x /= len_tmp;
    tmp.y /= len_tmp;
    tmp.z /= len_tmp;

    normal[i].x = tmp.x;
    normal[i].y = tmp.y;
    normal[i].z = tmp.z;
  }

  // output fiberDirection
  out.output_fiberDirection(fiber);
  vector<vec3>().swap(fiber);
  cout << "finish out fiber" << endl;

  // output sheetDirection
  out.output_sheetDirection(sheet);
  vector<vec3>().swap(sheet);
  cout << "finish out sheet" << endl;

  // output normalDirection
  out.output_normalDirection(normal);
  vector<vec3>().swap(normal);
  cout << "finish out normal" << endl;

  // // get edge_set
  // int num_edge_set;
  // vector<int> edge_set_id;
  // get_edge_set ges(verts, tet_edge_ids);
  // ges.edge_set(num_edge_set, edge_set_id);

  // // output edge_set
  // out.output_edge_set(num_edge_set, edge_set_id);
  // vector<int>().swap(tet_edge_ids);
  // vector<int>().swap(edge_set_id);
  // cout << "finish out edge_set" << endl;

  vector<int> sum_set;
  int cur_set_id = 1;
  int cur_sum_set = 0;
  cout << tets.size() << endl;
  for (int i = 0; i < tets.size(); i++) {
    if (tets[i].set_id == cur_set_id) {
      cur_sum_set++;
    } else {
      cur_set_id++;
      sum_set.push_back(cur_sum_set);
      cur_sum_set = 1;
    }
  }
  sum_set.push_back(cur_sum_set);

  out.output_sum_set(sum_set);
  vector<int>().swap(sum_set);
  cout << "finish out sum_set" << endl;

  // output tet_set
  out.output_tet_set(num_tet_set, tets_set_id);
  vector<tet>().swap(tets);
  vector<int>().swap(tets_set_id);
  cout << "finish out tet_set" << endl;

  // get bou_dirichlet
  for (size_t i = 0; i < nv; i++) {
    if (fabs(verts[i].x) < 0.001) {
      bou_dirichlet.push_back(1);
    } else {
      bou_dirichlet.push_back(0);
    }
  }

  out.output_bou_tag1(bou_dirichlet);

  vector<int>().swap(bou_dirichlet);
  vector<vert>().swap(verts);
  vector<tet>().swap(tets);

  out.output_end();

  return 0;
}
