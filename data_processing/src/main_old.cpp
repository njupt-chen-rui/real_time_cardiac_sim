#include<iostream>
#include<pbd_data/datastruct.h>
#include<pbd_data/read_tet.h>
#include<pbd_data/tetEdgeIds.h>
#include<pbd_data/tetSurfaceTriIds.h>
#include<pbd_data/edge_set.h>
#include<pbd_data/tet_set.h>
#include<pbd_data/bou_tag.h>
#include<pbd_data/output_mesh_data.h>
using namespace std;

int main(){
    cout << "begin" << endl;
    string file_name = "LV1";
    string file_format = ".inp";
    string input_path = "./data/input/" + file_name + file_format;
    string output_path = "./data/output/" + file_name + ".py";

    output_mesh_data out(output_path);
    // output name
    out.output_name(file_name);
    cout << "finish out name" << endl;

    vector<vert> verts;
    vector<tet> tets;
    // get verts and tetIds
    readinp_tet(input_path, verts, tets);

    // output verts
    out.output_verts(verts);
    cout << "finish out verts" << endl;

    // output tetIds
    out.output_tetIds(tets);
    cout << "finish out tetIds" << endl;

    // get tetEdgeIds
    vector<int> tet_edge_ids;
    tetEdgeIds(verts, tets, tet_edge_ids);

    // output tetEdgeIds
    out.output_tetEdgeIds(tet_edge_ids);
    cout << "finish out tetEdgeIds" << endl;

    // // get tetSurfaceTriIds
    // get_tet_surf gtst(verts, tets);
    // vector<int> tet_surface_tri_ids;
    // gtst.tetSurfaceTriIds(tet_surface_tri_ids);

    // // output tetSurfaceTriIds
    // out.output_tetSurfaceTriIds(tet_surface_tri_ids);
    // vector<int>().swap(tet_surface_tri_ids);
    // cout << "finish out tetSurfaceTriIds" << endl;

    // get fiberDirection 
    vector<vec3> fiber;
    for(size_t i = 0; i < tets.size(); i++){
        vec3 tmp = vec3{1., 0., 0.};
        fiber.push_back(tmp);
    }

    // output fiberDirection
    out.output_fiberDirection(fiber);
    vector<vec3>().swap(fiber);
    cout << "finish out fiber" << endl;

    // get sheetDirection
    vector<vec3> sheet;
    for(size_t i = 0; i < tets.size(); i++){
        vec3 tmp = vec3{0., 1., 0.};
        sheet.push_back(tmp);
    }

    // output sheetDirection
    out.output_sheetDirection(sheet);
    vector<vec3>().swap(sheet);
    cout << "finish out sheet" << endl;

    // get normalDirection
    vector<vec3> normal;
    for(size_t i = 0; i < tets.size(); i++){
        vec3 tmp = vec3{0., 0., 1.};
        normal.push_back(tmp);
    }

    // output normalDirection
    out.output_normalDirection(normal);
    vector<vec3>().swap(normal);
    cout << "finish out normal" << endl;

    // get edge_set
    int num_edge_set;
    vector<int> edge_set_id;
    get_edge_set ges(verts, tet_edge_ids);
    ges.edge_set(num_edge_set, edge_set_id);

    // output edge_set
    out.output_edge_set(num_edge_set, edge_set_id);
    vector<int>().swap(tet_edge_ids);
    vector<int>().swap(edge_set_id);
    cout << "finish out edge_set" << endl;

    // get tet_set
    int num_tet_set;
    vector<int> tets_set_id;
    get_tet_set gts(verts, tets);
    gts.tet_set_id(num_tet_set, tets_set_id);

    // output tet_set
    out.output_tet_set(num_tet_set, tets_set_id);
    vector<tet>().swap(tets);
    vector<int>().swap(tets_set_id);
    cout << "finish out tet_set" << endl;

    // get bou_tag_dirichlet
    vector<int> bou_tag1;
    get_bou_tag_1(verts, bou_tag1);

    // output bou_tag
    out.output_bou_tag1(bou_tag1);
    // vector<vert>().swap(verts);
    vector<int>().swap(bou_tag1);
    cout << "finish out bou_tag1" << endl;

    // get bou_tag_neumann
    vector<int> bou_tag2;
    get_bou_tag_2(verts, bou_tag2);

    // output bou_tag
    out.output_bou_tag2(bou_tag2);
    vector<vert>().swap(verts);
    vector<int>().swap(bou_tag2);
    cout << "finish out bou_tag2" << endl;

    out.output_end();
    
    return 0;
}