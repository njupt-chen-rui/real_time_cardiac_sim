import taichi as ti
import taichi.math as tm
import numpy as np
import project.Geometry as geo
import project.Electrophysiology as elec
import project.Dynamics as dyn
import project.GUI as gui
import argparse


def set_scene_whole_heart(args):

    # 导入数据
    from project.data.whole_heart import meshData
    body_name = meshData['name']
    geo_model, flag_dirichlet, flag_neumann = geo.read_body(meshData=meshData)
    num_per_tet_set_np = np.array(meshData['sum_tet_set'], dtype=int)

    # 是否使用各向异性
    dyn_model = dyn.Dynamics_XPBD_SNH_Active_aniso(body=geo_model, num_pts_np=num_per_tet_set_np, tag_dirichlet_all_dir=flag_dirichlet, tag_neumann=flag_neumann)
    save_path = args.save_path

    mygui = gui.Gui(geometry_model=geo_model, dynamics_model=dyn_model, body_name=body_name, save_path=save_path)
    mygui.interaction_operator.open_interaction_during_solving = args.dyn_ix  # 动态交互  

    # 场景配置
    mygui.set_electrophysiology_model_type("ap")
    apply_stimulation_Sinoatrial_Node(body=geo_model, tag_nid=1162, sti_val=1.5)
    geo_model.update_color_Vm()

    # 开启gui渲染
    mygui.display()


@ti.kernel
def apply_stimulation_Sinoatrial_Node(body: ti.template(), tag_nid: int, sti_val: float):
    for i in body.nodes:
        dis = (body.nodes[i][0] - body.nodes[tag_nid][0]) * (body.nodes[i][0] - body.nodes[tag_nid][0])\
              + (body.nodes[i][1] - body.nodes[tag_nid][1]) * (body.nodes[i][1] - body.nodes[tag_nid][1])\
              + (body.nodes[i][2] - body.nodes[tag_nid][2]) * (body.nodes[i][2] - body.nodes[tag_nid][2])
        dis = tm.sqrt(dis)
        if dis < 1.0:
            body.Vm[i] = sti_val
