import taichi as ti
import taichi.math as tm
import numpy as np
import project.Geometry as geo
import project.Electrophysiology as elec
import project.Dynamics as dyn
import project.GUI as gui
import argparse

# TODO: 
def set_scene_lv(args):

    # 导入数据
    from project.data.lv import meshData
    body_name = meshData['name']
    geo_model, flag_dirichlet, flag_neumann = geo.read_body(meshData=meshData)
    num_per_tet_set_np = np.array(meshData['sum_tet_set'], dtype=int)

    # 是否使用各向异性
    dyn_model = dyn.Dynamics_XPBD_SNH_Active(body=geo_model, num_pts_np=num_per_tet_set_np, tag_dirichlet_all_dir=flag_dirichlet, tag_neumann=flag_neumann)
    save_path = args.save_path

    mygui = gui.Gui(geometry_model=geo_model, dynamics_model=dyn_model, body_name=body_name, save_path=save_path)
    mygui.interaction_operator.open_interaction_during_solving = args.dyn_ix  # 动态交互  

    # 场景配置
    mygui.geometry_model.set_Ta(60.0)
    mygui.interaction_operator.dyn_op.Youngs_modulus = 1000.0
    mygui.interaction_operator.dyn_op.Poisson_ratio = 0.49
    mygui.interaction_operator.dt = 1.0 / 6.0
    mygui.interaction_operator.ele_op.open = False
    dyn_model.flag_update_Ta = False

    # 开启gui渲染
    mygui.display()
