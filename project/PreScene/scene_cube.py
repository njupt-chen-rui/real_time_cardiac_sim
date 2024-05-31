import taichi as ti
import taichi.math as tm
import numpy as np
import project.Geometry as geo
import project.Electrophysiology as elec
import project.Dynamics as dyn
import project.GUI as gui
import argparse


def set_scene_cube(args):
    from project.data.cube import meshData
    body_name = meshData['name']
    geo_model, flag_dirichlet, flag_neumann = geo.read_body(meshData=meshData)
    num_per_tet_set_np = np.array(meshData['sum_tet_set'], dtype=int)
    dyn_model = dyn.Dynamics_XPBD_SNH_Active_aniso(body=geo_model, num_pts_np=num_per_tet_set_np, tag_dirichlet_all_dir=flag_dirichlet, tag_neumann=flag_neumann)
    save_path = args.save_path

    mygui = gui.Gui(geometry_model=geo_model, dynamics_model=dyn_model, body_name=body_name, save_path=save_path)
    mygui.interaction_operator.ele_op.open = False  # 关闭电生理仿真
    mygui.interaction_operator.open_interaction_during_solving = args.dyn_ix  # 动态交互

    # 场景配置
    set_Vm(geo_model)
    geo_model.update_color_Vm_scene3()
    set_dirichlet_bou(geo_model=geo_model)
    get_fiber(geo_model)
    dyn_model.flag_update_Ta = False

    # 开启gui渲染
    mygui.display()


@ti.kernel
def set_dirichlet_bou(geo_model: ti.template()):
    for i in geo_model.nodes:
        if abs(geo_model.nodes[i][1]) < 1e-12:
            geo_model.bou_tag_dirichlet[i] = 1

@ti.kernel
def set_Vm(geo_model: ti.template()):
    tet = ti.static(geo_model.elements)
    for i in geo_model.Vm:
        geo_model.Vm[i] = geo_model.nodes[i][1] * 30.0
        geo_model.ver_Ta[i] = 0.5 * geo_model.Vm[i]

    for i in geo_model.elements:
        id0, id1, id2, id3 = tet[i][0], tet[i][1], tet[i][2], tet[i][3]
        geo_model.tet_Ta[i] = (geo_model.ver_Ta[id0] + geo_model.ver_Ta[id1] + geo_model.ver_Ta[id2] + geo_model.ver_Ta[id3]) / 4.0

@ti.kernel
def get_fiber(geo_model: ti.template()):
    for i in geo_model.tet_fiber:
        geo_model.tet_fiber[i] = tm.vec3(0.0, 1.0, 0.0)
        geo_model.tet_sheet[i] = tm.vec3(1.0, 0.0, 0.0)
