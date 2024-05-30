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

void read_vert_fiber(string path, vector<vec3> &vert_fiber, int num_verts){
    ifstream fin;
    fin.open(path);
    if (!fin.good()) {
        cout << "file open failed";
        exit(1);
    }
    for(int i = 0; i < num_verts; i++){
        // fin >> vert_fiber[i].x >> vert_fiber[i].y >> vert_fiber[i].z;
        fin >> vert_fiber[i].z >> vert_fiber[i].x >> vert_fiber[i].y;
    }
    fin.close();
}

vec3 vec3_cross(vec3 &a, vec3 &b){
    vec3 res;
    res.x = a.y * b.z - a.z * b.y;
    res.y = a.z * b.x - a.x * b.z;
    res.z = a.x * b.y - a.y * b.x;
    return res;
}

void readinp_test(string Path, vector<vert> &node) {
    ifstream fin;
    fin.open(Path);
    if (!fin.good()) {
        cout << "file open failed";
        exit(1);
    }
    double eps = 1e-9;
    string str, c;
    char ch;
    int a;
    double b1, b2, b3;
    vert tmp{0, 0, 0};
    cout<<"read node"<<endl;
    while(fin>>str){
        if(str=="*NODE") break;
    }
    bool flag = true;
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
        if ((fabs(tmp.x - node[a - 1].x) < eps && (fabs(tmp.x - node[a - 1].x) < eps) && (fabs(tmp.x - node[a - 1].x) < eps))){
            // cout << "sucess test" << endl;
        } else {
            flag = false;
            cout << "failed" <<endl;
            break;
        }
    }
    if(flag) cout <<"success test" << endl;
    return ;
}

void readinp_bou(string Path, vector<tri> &bou_base, vector<tri> &bou_endo_lv, vector<tri> &bou_epi_lv){
    cout << "begin read bou" << endl;
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
    cout<<"begin read node"<<endl;
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
    }

    cout<<"begin read base"<<endl;
    while(fin>>str){
        if(str.substr(0, 8)=="*ELEMENT") break;
    }
    while(1) {
        while(1){
            ch = fin.get();
            ch = fin.peek();
            if(ch == '*' || (ch <= '9' && ch >= '0')) break;
        }
        if(ch == '*') break;
        int triid;
        tri tmp_tri{0, 0, 0};
        fin >> triid >> c;
        fin >> tmp_tri.v[0] >> c;
        fin >> tmp_tri.v[1] >> c;
        fin >> tmp_tri.v[2];
        for (int i = 0; i < 3; i++) {
            tmp_tri.v[i] --;
        }
        bou_base.push_back(tmp_tri);
    }

    cout<<"begin read endo"<<endl;
    while(fin>>str){
        if(str.substr(0, 8)=="*ELEMENT") break;
    }
    while(1) {
        while(1){
            ch = fin.get();
            ch = fin.peek();
            if(ch == '*' || (ch <= '9' && ch >= '0')) break;
        }
        if(ch == '*') break;
        int triid;
        tri tmp_tri{0, 0, 0};
        fin >> triid >> c;
        fin >> tmp_tri.v[0] >> c;
        fin >> tmp_tri.v[1] >> c;
        fin >> tmp_tri.v[2];
        for (int i = 0; i < 3; i++) {
            tmp_tri.v[i] --;
        }
        bou_endo_lv.push_back(tmp_tri);
    }

    cout<<"begin read epi"<<endl;
    while(fin>>str){
        if(str.substr(0, 8)=="*ELEMENT") break;
    }
    while(1) {
        while(1){
            ch = fin.get();
            ch = fin.peek();
            if(ch == '*' || (ch <= '9' && ch >= '0')) break;
        }
        if(ch == '*') break;
        int triid;
        tri tmp_tri{0, 0, 0};
        fin >> triid >> c;
        fin >> tmp_tri.v[0] >> c;
        fin >> tmp_tri.v[1] >> c;
        fin >> tmp_tri.v[2];
        for (int i = 0; i < 3; i++) {
            tmp_tri.v[i] --;
        }
        bou_epi_lv.push_back(tmp_tri);
    }

    cout << "finish read bou" << endl;
    fin.close();
}

int main(){
    cout << "begin" << endl;
    string file_name = "LV1";
    string file_format = ".inp";
    string input_path = "./data/input/" + file_name + file_format;
    string output_path = "./data/output/" + file_name + "_test.py";

    output_mesh_data out(output_path);
    // output name
    out.output_name(file_name);
    cout << "finish out name" << endl;

    vector<vert> verts;
    vector<tet> tets;
    // get verts and tetIds
    readinp_tet(input_path, verts, tets);

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

    // // get tetSurfaceTriIds
    // get_tet_surf gtst(verts, tets);
    // vector<int> tet_surface_tri_ids;
    // gtst.tetSurfaceTriIds(tet_surface_tri_ids);

    // // output tetSurfaceTriIds
    // out.output_tetSurfaceTriIds(tet_surface_tri_ids);
    // vector<int>().swap(tet_surface_tri_ids);
    // cout << "finish out tetSurfaceTriIds" << endl;

    // get vert_fiber
    vector<vec3> vert_fiber;
    int num_vert = verts.size();
    vert_fiber.resize(num_vert);
    string vert_fiber_path = "./data/input/LV1_fiber.txt";
    read_vert_fiber(vert_fiber_path, vert_fiber, num_vert);

    vector<vec3> fiber, sheet, normal;
    int ne = tets.size();
    fiber.resize(ne);
    sheet.resize(ne);
    normal.resize(ne);
    for(int i = 0; i < ne; i++)
    {
        fiber[i].x = vert_fiber[tets[i].v[0]].x + vert_fiber[tets[i].v[1]].x + vert_fiber[tets[i].v[2]].x + vert_fiber[tets[i].v[3]].x;
        fiber[i].y = vert_fiber[tets[i].v[0]].y + vert_fiber[tets[i].v[1]].y + vert_fiber[tets[i].v[2]].y + vert_fiber[tets[i].v[3]].y;
        fiber[i].z = vert_fiber[tets[i].v[0]].z + vert_fiber[tets[i].v[1]].z + vert_fiber[tets[i].v[2]].z + vert_fiber[tets[i].v[3]].z;
        
        fiber[i].x /= 4.0;
        fiber[i].y /= 4.0;
        fiber[i].z /= 4.0;

        double len_f = sqrt(fiber[i].x * fiber[i].x + fiber[i].y * fiber[i].y + fiber[i].z * fiber[i].z);
        fiber[i].x /= len_f;
        fiber[i].y /= len_f;
        fiber[i].z /= len_f;

        if (fiber[i].z == 0 && fiber[i].y == 0){
            sheet[i] = vec3{0, 1, 0};
            normal[i] = vec3{0, 0, 1};
        } else if (fiber[i].z == 0) {
            sheet[i].x = 1.0; sheet[i].y = -fiber[i].x / fiber[i].y; sheet[i].z = 0;
            double len_s = sqrt(sheet[i].x * sheet[i].x + sheet[i].y * sheet[i].y + sheet[i].z * sheet[i].z);
            sheet[i].x /= len_s;
            sheet[i].y /= len_s;
            sheet[i].z /= len_s;
            normal[i] = vec3_cross(fiber[i], sheet[i]);
        } else {
            sheet[i].x = 1.0; sheet[i].y = 0; sheet[i].z = -fiber[i].x / fiber[i].z;
            double len_s = sqrt(sheet[i].x * sheet[i].x + sheet[i].y * sheet[i].y + sheet[i].z * sheet[i].z);
            sheet[i].x /= len_s;
            sheet[i].y /= len_s;
            sheet[i].z /= len_s;
            normal[i] = vec3_cross(fiber[i], sheet[i]);
        }
    }

    // output vertex fiber
    out.output_vert_fiber(vert_fiber);
    vector<vec3>().swap(vert_fiber);
    cout << "finish out vert fiber" << endl;

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
    vector<int>().swap(tets_set_id);
    cout << "finish out tet_set" << endl;

    // get bou_tag_dirichlet
    vector<int> bou_tag1;
    get_bou_tag_1(verts, bou_tag1);

    // output bou_tag
    out.output_bou_tag1(bou_tag1);
    vector<int>().swap(bou_tag1);
    cout << "finish out bou_tag1" << endl;

    // get bou_tag_neumann
    vector<int> bou_tag2;
    get_bou_tag_2(verts, bou_tag2);

    // output bou_tag
    out.output_bou_tag2(bou_tag2);
    vector<int>().swap(bou_tag2);
    cout << "finish out bou_tag2" << endl;

    // read bou of LV1
    string LV1_bou_path = "./data/input/bou_LV1.inp";
    // readinp_test(LV1_bou_path, verts);
    vector<tri> bou_base, bou_endo_lv, bou_epi_lv;
    readinp_bou(LV1_bou_path, bou_base, bou_endo_lv, bou_epi_lv);

    out.output_bou_endo_lv_face(bou_endo_lv);
    out.output_bou_epi_lv_face(bou_epi_lv);
    cout << "finish out bou endo lv and bou epi lv" << endl;
    vector<tri>().swap(bou_base);
    vector<tri>().swap(bou_endo_lv);
    vector<tri>().swap(bou_epi_lv);

    out.output_end();
    vector<vert>().swap(verts);
    vector<tet>().swap(tets);
    return 0;
}
