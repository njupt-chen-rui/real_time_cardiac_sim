#include<vector>
#include<set>
#include<pbd_data/datastruct.h>
using namespace std;

struct get_tet_surf{
    const vector<vert> &node;
    vert gc;
    const vector<tet> &tets;
    struct tri{
        int a, b, c;
        bool operator < (const tri other)const{
            if(a == other.a){
                if(b == other.b){
                    return c < other.c;
                }
                return b < other.b;
            }
            return a < other.a;
        }
    };
    set<tri> se;
    set<tri>::iterator it;
    
    bool check_out_surface(int p1, int p2, int p3){
        struct vert gc_p1, p1_p2, p1_p3, normal;
        double theta;
        gc_p1.x = node[p1].x - gc.x;
        gc_p1.y = node[p1].y - gc.y;
        gc_p1.z = node[p1].z - gc.z;
        p1_p2.x = node[p2].x - node[p1].x;
        p1_p2.y = node[p2].y - node[p1].y;
        p1_p2.z = node[p2].z - node[p1].z;
        p1_p3.x = node[p3].x - node[p1].x;
        p1_p3.y = node[p3].y - node[p1].y;
        p1_p3.z = node[p3].z - node[p1].z;
        normal.x = p1_p2.y * p1_p3.z - p1_p3.y * p1_p2.z;
        normal.y = p1_p3.x * p1_p2.z - p1_p2.x * p1_p3.z;
        normal.z = p1_p2.x * p1_p3.y - p1_p3.x * p1_p2.y;
        theta = gc_p1.x * normal.x + gc_p1.y * normal.y + gc_p1.z * normal.z;
        if(theta >= 0.0){
            return true;
        }else return false;
    }

    void add_out_surface(int p1,int p2,int p3){
        int a, b, c;
        if(p1<p2&&p1<p3){
            a = p1; b = p2; c = p3;
        }else if(p2<p1&&p2<p3){
            a = p2; b = p3; c = p1;
        }else{
            a = p3; b = p1; c = p2;
        }
        bool flag = true;
        for(it = se.begin();it!=se.end();){
            if((*it).a==a&&(*it).b==b&&(*it).c==c){
                se.erase(it++);
                flag=false;
            }else if((*it).a==a&&(*it).c==b&&(*it).b==c){
                se.erase(it++);
                flag=false;
            }else{
                it++;
            }
        }
        if(flag){
            struct tri tmp;
            tmp.a = a; tmp.b = b; tmp.c = c;
            se.insert(tmp);
        }
    }

    void tetSurfaceTriIds(vector<int> &res){
        int NV = node.size();
        int n = tets.size();
        for(int i=0;i<n;i++){
            int a = tets[i].v[0];
            int b = tets[i].v[1];
            int c = tets[i].v[2];
            int d = tets[i].v[3];
            gc.x = (node[a].x + node[b].x + node[c].x + node[d].x) / 4.0;
            gc.y = (node[a].y + node[b].y + node[c].y + node[d].y) / 4.0;
            gc.z = (node[a].z + node[b].z + node[c].z + node[d].z) / 4.0;
            if(check_out_surface(a,b,c)){
                add_out_surface(a,b,c);
            }else{
                add_out_surface(a,c,b);
            }
            if(check_out_surface(a,c,d)){
                add_out_surface(a,c,d);
            }else{
                add_out_surface(a,d,c);
            }
            if(check_out_surface(a,b,d)){
                add_out_surface(a,b,d);
            }else{
                add_out_surface(a,d,b);
            }
            if(check_out_surface(b,c,d)){
                add_out_surface(b,c,d);
            }else{
                add_out_surface(b,d,c);
            }
        }
        for(it = se.begin(); it!=se.end(); it++){
            res.push_back((*it).a);
            res.push_back((*it).b);
            res.push_back((*it).c);
        }

        se.clear();
    }
    
    get_tet_surf(const vector<vert> &verts, const vector<tet> &tets): node{verts}, tets{tets}{}
};
