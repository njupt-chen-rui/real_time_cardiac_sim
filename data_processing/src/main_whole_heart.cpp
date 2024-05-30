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

void read_node_from_tetgen(string input_path, vector<vert>& nodes) {
  ifstream fin;
  fin.open(input_path);
  if (!fin.good()) {
    cout << "file open failed" << endl;
    exit(1);
  }
  cout << "begin read nodes" << endl;
  int num_nodes, dim, feature, boudary;
  fin >> num_nodes >> dim >> feature >> boudary;
  int id_nodes;
  vert node_pos{0, 0, 0};
  for (int i = 0; i < num_nodes; i++) {
    fin >> id_nodes >> node_pos.x >> node_pos.y >> node_pos.z;
    nodes.emplace_back(node_pos);
  }
  cout << "finish read nodes" << endl;
  fin.close();
}

void read_ele_from_tetgen(string input_path, vector<tet>& tets) {
  ifstream fin;
  fin.open(input_path);
  if (!fin.good()) {
    cout << "file open failed" << endl;
    exit(1);
  }
  cout << "begin read elements" << endl;
  int num_elements, nodes_per_elements, feature;
  fin >> num_elements >> nodes_per_elements >> feature;
  int id_elements;
  tet nodes_id_of_element{0, 0, 0, 0};
  for (int i = 0; i < num_elements; i++) {
    fin >> id_elements >> nodes_id_of_element.v[0] >>
        nodes_id_of_element.v[1] >> nodes_id_of_element.v[2] >>
        nodes_id_of_element.v[3];
    tets.emplace_back(nodes_id_of_element);
  }
  cout << "finish read elements" << endl;
  fin.close();
}

void read_fiber_from_txt(string path, int nv, vector<vec3>& fiber,
                         vector<vec3>& sheet, vector<vec3>& normal) {
  ifstream fin;
  fin.open(path);
  if (!fin.good()) {
    cout << "fiber file open failed";
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

vec3 vec3_cross(vec3& a, vec3& b) {
  vec3 res;
  res.x = a.y * b.z - a.z * b.y;
  res.y = a.z * b.x - a.x * b.z;
  res.z = a.x * b.y - a.y * b.x;
  return res;
}

int main() {
  cout << "begin" << endl;

  // 设置读入写出文件
  string file_name = "whole_heart";
  string input_path_node = "./data/input/whole_heart/modelB.node";
  string input_path_elem = "./data/input/whole_heart/modelB.ele";
  string output_path = "./data/output/whole_heart.py";

  // 输出文件头
  output_mesh_data out(output_path);

  // 输出模型名字
  out.output_name(file_name);
  cout << "finish out name" << endl;

  // 申请内存
  vector<vert> verts;
  vector<tet> tets;

  // 读顶点数据
  read_node_from_tetgen(input_path_node, verts);
  int nv = verts.size();
  // 读单元数据
  read_ele_from_tetgen(input_path_elem, tets);
  int ne = tets.size();

  // 输出顶点 verts
  out.output_verts(verts);
  cout << "finish out verts" << endl;

  // 划分网格独立集合
  // get tet_set
  int num_tet_set;          // 集合数量
  vector<int> tets_set_id;  // 网格i所属的集合id
  get_tet_set gts(verts, tets);
  gts.tet_set_id(num_tet_set, tets_set_id);
  for (int i = 0; i < tets.size(); i++) {
    tets[i].set_id = tets_set_id[i];
  }
  sort(tets.begin(), tets.end());
  sort(tets_set_id.begin(), tets_set_id.end());

  // 输出网格单元
  out.output_tetIds(tets);
  cout << "finish out tetIds" << endl;

  // 纤维方向
  // 读取顶点纤维方向
  string fiber_file_name = "./data/input/whole_heart/modelB.txt";
  vector<vec3> vert_fiber;
  vector<vec3> vert_sheet;
  vector<vec3> vert_normal;
  vert_fiber.resize(nv);
  vert_sheet.resize(nv);
  vert_normal.resize(nv);
  read_fiber_from_txt(fiber_file_name, nv, vert_fiber, vert_sheet, vert_normal);

  vector<vec3> fiber, sheet, normal;
  fiber.resize(ne);
  sheet.resize(ne);
  normal.resize(ne);
  double eps = 1e-12;
  for (int i = 0; i < ne; i++) {
    fiber[i].x = vert_fiber[tets[i].v[0]].x + vert_fiber[tets[i].v[1]].x +
                 vert_fiber[tets[i].v[2]].x + vert_fiber[tets[i].v[3]].x;
    fiber[i].y = vert_fiber[tets[i].v[0]].y + vert_fiber[tets[i].v[1]].y +
                 vert_fiber[tets[i].v[2]].y + vert_fiber[tets[i].v[3]].y;
    fiber[i].z = vert_fiber[tets[i].v[0]].z + vert_fiber[tets[i].v[1]].z +
                 vert_fiber[tets[i].v[2]].z + vert_fiber[tets[i].v[3]].z;

    sheet[i].x = vert_sheet[tets[i].v[0]].x + vert_sheet[tets[i].v[1]].x +
                 vert_sheet[tets[i].v[2]].x + vert_sheet[tets[i].v[3]].x;
    sheet[i].y = vert_sheet[tets[i].v[0]].y + vert_sheet[tets[i].v[1]].y +
                 vert_sheet[tets[i].v[2]].y + vert_sheet[tets[i].v[3]].y;
    sheet[i].z = vert_sheet[tets[i].v[0]].z + vert_sheet[tets[i].v[1]].z +
                 vert_sheet[tets[i].v[2]].z + vert_sheet[tets[i].v[3]].z;

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

  // 输出fiberDirection
  out.output_fiberDirection(fiber);
  vector<vec3>().swap(vert_fiber);
  vector<vec3>().swap(fiber);
  cout << "finish out fiber" << endl;

  // 输出sheetDirection
  out.output_sheetDirection(sheet);
  vector<vec3>().swap(vert_sheet);
  vector<vec3>().swap(sheet);
  cout << "finish out sheet" << endl;

  // 输出normalDirection
  out.output_normalDirection(normal);
  vector<vec3>().swap(vert_normal);
  vector<vec3>().swap(normal);
  cout << "finish out normal" << endl;

  // sum_set为每个网格集合中包含的网格数量
  vector<int> sum_set;
  int cur_set_id = 1;
  int cur_sum_set = 0;
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
  // 网格所属的集合的索引
  out.output_tet_set(num_tet_set, tets_set_id);
  vector<int>().swap(tets_set_id);
  cout << "finish out tet_set" << endl;

  // 输出边界
  // out.output_bou_base_face(bou_base);
  // out.output_bou_endo_lv_face(bou_endo_lv);
  // out.output_bou_endo_rv_face(bou_endo_rv);
  // out.output_bou_epi_face(bou_epi);
  // cout << "finish out bou endo, epi and base" << endl;

  // 输出文件尾
  out.output_end();

  // 释放内存
  vector<int>().swap(tets_set_id);
  vector<vert>().swap(verts);
  vector<tet>().swap(tets);

  cout << "finish all" << endl;

  return 0;
}
