#include<iostream>
#include<pbd_data/datastruct.h>
#include<pbd_data/read_tet.h>
#include<pbd_data/tetEdgeIds.h>
#include<pbd_data/tetSurfaceTriIds.h>
#include<pbd_data/edge_set.h>
#include<pbd_data/tet_set.h>
#include<pbd_data/bou_tag.h>
#include<pbd_data/output_mesh_data.h>
#include<pbd_data/test.h>
using namespace std;

int main()
{
    cout << "begin" << endl;
    string file_name = "heart_mesh";
    string file_format = ".py";
    string input_path = "./data/input/" + file_name + "_origin" + file_format;
    string output_path = "./data/output/" + file_name + ".py";

    output_mesh_data out(output_path);
    // output name
    out.output_name(file_name);
    cout << "finish out name" << endl;

    // get verts and tetIds
    vector<vert> verts;
    vector<tet> tets;
    // get fiberDirection 
    vector<vec3> vert_fiber, fiber;
    // get sheetDirection
    vector<vec3> vert_sheet, sheet;
    // get normalDirection
    vector<vec3> vert_normal, normal;
    readpy_tet(input_path, verts, tets, vert_fiber, vert_sheet, vert_normal);
    // test_vert(verts);
    // test_tet(tets);

    // output verts
    out.output_verts(verts);
    cout << "finish out verts" << endl;

    // get tet_set
    int num_tet_set;
    vector<int> tets_set_id;
    get_tet_set gts(verts, tets);
    gts.tet_set_id(num_tet_set, tets_set_id);
    for(int i=0;i<tets.size();i++){
        tets[i].set_id = tets_set_id[i];
    }
    sort(tets.begin(), tets.end());
    sort(tets_set_id.begin(), tets_set_id.end());

    // output tetIds
    out.output_tetIds(tets);
    cout << "finish out tetIds" << endl;

    // get tetEdgeIds
    vector<int> tet_edge_ids;
    tetEdgeIds(verts, tets, tet_edge_ids);

    // output tetEdgeIds
    out.output_tetEdgeIds(tet_edge_ids);
    cout << "finish out tetEdgeIds" << endl;


    //sample fiber, sheet, normal from vert to tet
    int ne = tets.size();
    fiber.resize(ne);
    sheet.resize(ne);
    normal.resize(ne);
    for(int i = 0; i < ne; i++)
    {
        fiber[i].x = vert_fiber[tets[i].v[0]].x + vert_fiber[tets[i].v[1]].x + vert_fiber[tets[i].v[2]].x + vert_fiber[tets[i].v[3]].x;
        sheet[i].x = vert_sheet[tets[i].v[0]].x + vert_sheet[tets[i].v[1]].x + vert_sheet[tets[i].v[2]].x + vert_sheet[tets[i].v[3]].x;
        normal[i].x = vert_normal[tets[i].v[0]].x + vert_normal[tets[i].v[1]].x + vert_normal[tets[i].v[2]].x + vert_normal[tets[i].v[3]].x;

        fiber[i].y = vert_fiber[tets[i].v[0]].y + vert_fiber[tets[i].v[1]].y + vert_fiber[tets[i].v[2]].y + vert_fiber[tets[i].v[3]].y;
        sheet[i].y = vert_sheet[tets[i].v[0]].y + vert_sheet[tets[i].v[1]].y + vert_sheet[tets[i].v[2]].y + vert_sheet[tets[i].v[3]].y;
        normal[i].y = vert_normal[tets[i].v[0]].y + vert_normal[tets[i].v[1]].y + vert_normal[tets[i].v[2]].y + vert_normal[tets[i].v[3]].y;
        
        fiber[i].z = vert_fiber[tets[i].v[0]].z + vert_fiber[tets[i].v[1]].z + vert_fiber[tets[i].v[2]].z + vert_fiber[tets[i].v[3]].z;
        sheet[i].z = vert_sheet[tets[i].v[0]].z + vert_sheet[tets[i].v[1]].z + vert_sheet[tets[i].v[2]].z + vert_sheet[tets[i].v[3]].z;
        normal[i].z = vert_normal[tets[i].v[0]].z + vert_normal[tets[i].v[1]].z + vert_normal[tets[i].v[2]].z + vert_normal[tets[i].v[3]].z;
        
        fiber[i].x /= 4.0;
        sheet[i].x /= 4.0;
        normal[i].x /= 4.0;

        fiber[i].y /= 4.0;
        sheet[i].y /= 4.0;
        normal[i].y /= 4.0;

        fiber[i].z /= 4.0;
        sheet[i].z /= 4.0;
        normal[i].z /= 4.0;

        double len_f = sqrt(fiber[i].x * fiber[i].x + fiber[i].y * fiber[i].y + fiber[i].z * fiber[i].z);
        double len_s = sqrt(sheet[i].x * sheet[i].x + sheet[i].y * sheet[i].y + sheet[i].z * sheet[i].z);
        double len_n = sqrt(normal[i].x * normal[i].x + normal[i].y * normal[i].y + normal[i].z * normal[i].z);
        // cout << len_f << " " << len_s << " " << len_n << endl;

        fiber[i].x /= len_f;
        sheet[i].x /= len_s;
        normal[i].x /= len_n;

        fiber[i].y /= len_f;
        sheet[i].y /= len_s;
        normal[i].y /= len_n;

        fiber[i].z /= len_f;
        sheet[i].z /= len_s;
        normal[i].z /= len_n;

    }

// test fiber, sheet, normal dir
{
    // for(int i=0;i<fiber.size();i++){
    //     cout << fiber[i].x << " " << fiber[i].y << " " << fiber[i].z << endl;
    //     cout << sqrt(fiber[i].x * fiber[i].x + fiber[i].y * fiber[i].y + fiber[i].z * fiber[i].z)<<endl;
    // }
    // for(int i=0;i<sheet.size();i++){
    //     cout << sheet[i].x << " " << sheet[i].y << " " << sheet[i].z << endl;
    //     cout << sqrt(sheet[i].x * sheet[i].x + sheet[i].y * sheet[i].y + sheet[i].z * sheet[i].z)<<endl;
    // }
    // for(int i=0;i<normal.size();i++){
    //     cout << normal[i].x << " " << normal[i].y << " " << normal[i].z << endl;
    //     cout << sqrt(normal[i].x * normal[i].x + normal[i].y * normal[i].y + normal[i].z * normal[i].z)<<endl;
    // }

    // cout << verts.size() << " " << tets.size() << endl;
    // cout << fiber.size() << " " << sheet.size() << " " << normal.size() << endl;
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

    vector<int> sum_set;
    int cur_set_id = 1;
    int cur_sum_set = 0;
    cout << tets.size() << endl;
    for(int i=0;i<tets.size();i++){
        // cout<< tets[i].set_id << endl;
        if(tets[i].set_id == cur_set_id){
            cur_sum_set++;
        }else{
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