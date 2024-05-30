#pragma once
#include <math.h>
#include <memory.h>
#include <pbd_data/datastruct.h>

#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <vector>

using namespace std;

struct output_mesh_data {
  string out_path;
  ofstream fout;

  output_mesh_data(string out_path) : out_path{out_path} {
    fout.open(out_path);
    // fout.open(out_path, ios::app);
    fout << "meshData = {" << endl;
  }

  ~output_mesh_data() { fout.close(); }

  void output_name(string name) {
    fout << "\t'name': \"" + name + "\"," << endl;
  }

  void output_verts(vector<vert> &verts) {
    fout << "\t'verts': [" << endl;
    int cnt = 0;
    for (size_t i = 0; i < verts.size(); i++) {
      if (cnt == 0) fout << "\t\t";
      fout << verts[i].x << ", " << verts[i].y << ", " << verts[i].z;
      if (i != verts.size() - 1) {
        fout << ", ";
      } else {
        fout << endl;
        fout << "\t]," << endl;
        break;
      }
      cnt++;
      if (cnt == 10) {
        cnt = 0;
        fout << endl;
      }
    }
  }

  // vert_id in .py file = vert_id in .inp file - 1
  void output_tetIds(vector<tet> &tetIds) {
    fout << "\t'tetIds': [" << endl;
    int cnt = 0;
    for (size_t i = 0; i < tetIds.size(); i++) {
      if (cnt == 0) fout << "\t\t";
      fout << tetIds[i].v[0] << ", " << tetIds[i].v[1] << ", " << tetIds[i].v[2]
           << ", " << tetIds[i].v[3];
      if (i != tetIds.size() - 1) {
        fout << ", ";
      } else {
        fout << endl;
        fout << "\t]," << endl;
        break;
      }
      cnt++;
      if (cnt == 10) {
        cnt = 0;
        fout << endl;
      }
    }
  }

  void output_tetEdgeIds(vector<int> &tetedgeids) {
    fout << "\t'tetEdgeIds': [" << endl;
    int cnt = 0;
    for (size_t i = 0; i < tetedgeids.size() / 2; i++) {
      if (cnt == 0) fout << "\t\t";
      fout << tetedgeids[i * 2] << ", " << tetedgeids[i * 2 + 1];
      if (i != tetedgeids.size() / 2 - 1) {
        fout << ", ";
      } else {
        fout << endl;
        fout << "\t]," << endl;
        break;
      }
      cnt++;
      if (cnt == 10) {
        cnt = 0;
        fout << endl;
      }
    }
  }

  void output_tetSurfaceTriIds(vector<int> &tet_surf_tri_ids) {
    fout << "\t'tetSurfaceTriIds': [" << endl;
    int cnt = 0;
    for (size_t i = 0; i < tet_surf_tri_ids.size() / 3; i++) {
      if (cnt == 0) fout << "\t\t";
      fout << tet_surf_tri_ids[i * 3] << ", " << tet_surf_tri_ids[i * 3 + 1]
           << ", " << tet_surf_tri_ids[i * 3 + 2];
      if (i != tet_surf_tri_ids.size() / 3 - 1) {
        fout << ", ";
      } else {
        fout << endl;
        fout << "\t]," << endl;
        break;
      }
      cnt++;
      if (cnt == 10) {
        cnt = 0;
        fout << endl;
      }
    }
  }

  void output_vert_fiber(vector<vec3> &vert_fiber) {
    fout << "\t'vert_fiber': [" << endl;
    int cnt = 0;
    for (size_t i = 0; i < vert_fiber.size(); i++) {
      if (cnt == 0) fout << "\t\t";
      fout << vert_fiber[i].x << ", " << vert_fiber[i].y << ", "
           << vert_fiber[i].z;
      if (i != vert_fiber.size() - 1) {
        fout << ", ";
      } else {
        fout << endl;
        fout << "\t]," << endl;
        break;
      }
      cnt++;
      if (cnt == 10) {
        cnt = 0;
        fout << endl;
      }
    }
  }

  void output_fiberDirection(vector<vec3> &fiber) {
    fout << "\t'fiberDirection': [" << endl;
    int cnt = 0;
    for (size_t i = 0; i < fiber.size(); i++) {
      if (cnt == 0) fout << "\t\t";
      fout << fiber[i].x << ", " << fiber[i].y << ", " << fiber[i].z;
      if (i != fiber.size() - 1) {
        fout << ", ";
      } else {
        fout << endl;
        fout << "\t]," << endl;
        break;
      }
      cnt++;
      if (cnt == 10) {
        cnt = 0;
        fout << endl;
      }
    }
  }

  void output_sheetDirection(vector<vec3> &sheet) {
    fout << "\t'sheetDirection': [" << endl;
    int cnt = 0;
    for (size_t i = 0; i < sheet.size(); i++) {
      if (cnt == 0) fout << "\t\t";
      fout << sheet[i].x << ", " << sheet[i].y << ", " << sheet[i].z;
      if (i != sheet.size() - 1) {
        fout << ", ";
      } else {
        fout << endl;
        fout << "\t]," << endl;
        break;
      }
      cnt++;
      if (cnt == 10) {
        cnt = 0;
        fout << endl;
      }
    }
  }

  void output_normalDirection(vector<vec3> &normal) {
    fout << "\t'normalDirection': [" << endl;
    int cnt = 0;
    for (size_t i = 0; i < normal.size(); i++) {
      if (cnt == 0) fout << "\t\t";
      fout << normal[i].x << ", " << normal[i].y << ", " << normal[i].z;
      if (i != normal.size() - 1) {
        fout << ", ";
      } else {
        fout << endl;
        fout << "\t]," << endl;
        break;
      }
      cnt++;
      if (cnt == 10) {
        cnt = 0;
        fout << endl;
      }
    }
  }

  void output_edge_set(int &num_edge_set, vector<int> &edge_set_id) {
    fout << "\t'num_edge_set': [" << num_edge_set << "]," << endl;

    fout << "\t'edge_set': [" << endl;
    int cnt = 0;
    for (size_t i = 0; i < edge_set_id.size(); i++) {
      if (cnt == 0) fout << "\t\t";
      fout << edge_set_id[i];
      if (i != edge_set_id.size() - 1) {
        fout << ", ";
      } else {
        fout << endl;
        fout << "\t]," << endl;
        break;
      }
      cnt++;
      if (cnt == 10) {
        cnt = 0;
        fout << endl;
      }
    }
  }

  void output_tet_set(int &num_tet_set, vector<int> &tets_set_id) {
    fout << "\t'num_tet_set': [" << num_tet_set << "]," << endl;

    fout << "\t'tet_set': [" << endl;
    int cnt = 0;
    for (size_t i = 0; i < tets_set_id.size(); i++) {
      if (cnt == 0) fout << "\t\t";
      fout << tets_set_id[i];
      if (i != tets_set_id.size() - 1) {
        fout << ", ";
      } else {
        fout << endl;
        fout << "\t]," << endl;
        break;
      }
      cnt++;
      if (cnt == 10) {
        cnt = 0;
        fout << endl;
      }
    }
  }

  void output_sum_set(vector<int> &sum_set) {
    fout << "\t'sum_tet_set': [" << endl;
    int cnt = 0;
    for (size_t i = 0; i < sum_set.size(); i++) {
      if (cnt == 0) fout << "\t\t";
      fout << sum_set[i];
      if (i != sum_set.size() - 1) {
        fout << ", ";
      } else {
        fout << endl;
        fout << "\t]," << endl;
        break;
      }
      cnt++;
      if (cnt == 10) {
        cnt = 0;
        fout << endl;
      }
    }
  }

  void output_bou_tag1(vector<int> &bou_tag) {
    fout << "\t'bou_tag_dirichlet': [" << endl;
    int cnt = 0;
    for (size_t i = 0; i < bou_tag.size(); i++) {
      if (cnt == 0) fout << "\t\t";
      fout << bou_tag[i];
      if (i != bou_tag.size() - 1) {
        fout << ", ";
      } else {
        fout << endl;
        fout << "\t]," << endl;
        break;
      }
      cnt++;
      if (cnt == 10) {
        cnt = 0;
        fout << endl;
      }
    }
  }

  void output_bou_tag2(vector<int> &bou_tag) {
    fout << "\t'bou_tag_neumann': [" << endl;
    int cnt = 0;
    for (size_t i = 0; i < bou_tag.size(); i++) {
      if (cnt == 0) fout << "\t\t";
      fout << bou_tag[i];
      if (i != bou_tag.size() - 1) {
        fout << ", ";
      } else {
        fout << endl;
        fout << "\t]," << endl;
        break;
      }
      cnt++;
      if (cnt == 10) {
        cnt = 0;
        fout << endl;
      }
    }
  }

  void output_bou_endo_lv_face(vector<tri> &bou_endo_lv) {
    fout << "\t'bou_endo_lv_face': [" << endl;
    int cnt = 0;
    for (size_t i = 0; i < bou_endo_lv.size(); i++) {
      if (cnt == 0) fout << "\t\t";
      fout << bou_endo_lv[i].v[0] << ", " << bou_endo_lv[i].v[1] << ", "
           << bou_endo_lv[i].v[2];
      if (i != bou_endo_lv.size() - 1) {
        fout << ", ";
      } else {
        fout << endl;
        fout << "\t]," << endl;
        break;
      }
      cnt++;
      if (cnt == 10) {
        cnt = 0;
        fout << endl;
      }
    }
  }

  void output_bou_endo_rv_face(vector<tri> &bou_endo_rv) {
    fout << "\t'bou_endo_rv_face': [" << endl;
    int cnt = 0;
    for (size_t i = 0; i < bou_endo_rv.size(); i++) {
      if (cnt == 0) fout << "\t\t";
      fout << bou_endo_rv[i].v[0] << ", " << bou_endo_rv[i].v[1] << ", "
           << bou_endo_rv[i].v[2];
      if (i != bou_endo_rv.size() - 1) {
        fout << ", ";
      } else {
        fout << endl;
        fout << "\t]," << endl;
        break;
      }
      cnt++;
      if (cnt == 10) {
        cnt = 0;
        fout << endl;
      }
    }
  }

  void output_bou_epi_lv_face(vector<tri> &bou_epi_lv) {
    fout << "\t'bou_epi_lv_face': [" << endl;
    int cnt = 0;
    for (size_t i = 0; i < bou_epi_lv.size(); i++) {
      if (cnt == 0) fout << "\t\t";
      fout << bou_epi_lv[i].v[0] << ", " << bou_epi_lv[i].v[1] << ", "
           << bou_epi_lv[i].v[2];
      if (i != bou_epi_lv.size() - 1) {
        fout << ", ";
      } else {
        fout << endl;
        fout << "\t]," << endl;
        break;
      }
      cnt++;
      if (cnt == 10) {
        cnt = 0;
        fout << endl;
      }
    }
  }

  void output_bou_epi_face(vector<tri> &bou_epi) {
    fout << "\t'bou_epi_face': [" << endl;
    int cnt = 0;
    for (size_t i = 0; i < bou_epi.size(); i++) {
      if (cnt == 0) fout << "\t\t";
      fout << bou_epi[i].v[0] << ", " << bou_epi[i].v[1] << ", "
           << bou_epi[i].v[2];
      if (i != bou_epi.size() - 1) {
        fout << ", ";
      } else {
        fout << endl;
        fout << "\t]," << endl;
        break;
      }
      cnt++;
      if (cnt == 10) {
        cnt = 0;
        fout << endl;
      }
    }
  }

  void output_bou_base_face(vector<tri> &bou_base) {
    fout << "\t'bou_base_face': [" << endl;
    int cnt = 0;
    for (size_t i = 0; i < bou_base.size(); i++) {
      if (cnt == 0) fout << "\t\t";
      fout << bou_base[i].v[0] << ", " << bou_base[i].v[1] << ", "
           << bou_base[i].v[2];
      if (i != bou_base.size() - 1) {
        fout << ", ";
      } else {
        fout << endl;
        fout << "\t]," << endl;
        break;
      }
      cnt++;
      if (cnt == 10) {
        cnt = 0;
        fout << endl;
      }
    }
  }

  void output_end() { fout << "}" << endl; }
};