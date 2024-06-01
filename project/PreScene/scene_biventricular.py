import taichi as ti
import taichi.math as tm
import numpy as np
import project.Geometry as geo
import project.Electrophysiology as elec
import project.Dynamics as dyn
import project.GUI as gui
import argparse


def set_scene_biventricular_ep_free_pulse(args):

    # 导入数据
    from project.data.biventricular import meshData
    body_name = meshData['name']
    geo_model, flag_dirichlet, flag_neumann = geo.read_body(meshData=meshData)
    num_per_tet_set_np = np.array(meshData['sum_tet_set'], dtype=int)

    # 是否使用各向异性
    # dyn_model = dyn.Dynamics_XPBD_SNH_Active_aniso(body=geo_model, num_pts_np=num_per_tet_set_np, tag_dirichlet_all_dir=flag_dirichlet, tag_neumann=flag_neumann)
    dyn_model = dyn.Dynamics_XPBD_SNH_Active(body=geo_model, num_pts_np=num_per_tet_set_np, tag_dirichlet_all_dir=flag_dirichlet, tag_neumann=flag_neumann)
    save_path = args.save_path

    mygui = gui.Gui(geometry_model=geo_model, dynamics_model=dyn_model, body_name=body_name, save_path=save_path)
    mygui.interaction_operator.open_interaction_during_solving = args.dyn_ix  # 动态交互  

    # 场景配置
    mygui.set_electrophysiology_model_type("ap")
    mygui.interaction_operator.dyn_op.open = False
    # geo_model.update_color_Vm()

    # 开启gui渲染
    mygui.display()

def set_scene_biventricular_ep_scroll_pulse(args):

    # 导入数据
    from project.data.biventricular import meshData
    body_name = meshData['name']
    geo_model, flag_dirichlet, flag_neumann = geo.read_body(meshData=meshData)
    num_per_tet_set_np = np.array(meshData['sum_tet_set'], dtype=int)

    # 是否使用各向异性
    # dyn_model = dyn.Dynamics_XPBD_SNH_Active_aniso(body=geo_model, num_pts_np=num_per_tet_set_np, tag_dirichlet_all_dir=flag_dirichlet, tag_neumann=flag_neumann)
    dyn_model = dyn.Dynamics_XPBD_SNH_Active(body=geo_model, num_pts_np=num_per_tet_set_np, tag_dirichlet_all_dir=flag_dirichlet, tag_neumann=flag_neumann)
    save_path = args.save_path

    mygui = gui.Gui(geometry_model=geo_model, dynamics_model=dyn_model, body_name=body_name, save_path=save_path)
    mygui.interaction_operator.open_interaction_during_solving = args.dyn_ix  # 动态交互  

    # 场景配置
    mygui.set_electrophysiology_model_type("ap")
    mygui.interaction_operator.dyn_op.open = False
    # geo_model.update_color_Vm()

    # 开启gui渲染
    mygui.display()

def set_scene_biventricular_em_free_pulse(args):

    # 导入数据
    from project.data.biventricular import meshData
    body_name = meshData['name']
    geo_model, flag_dirichlet, flag_neumann = geo.read_body(meshData=meshData)
    num_per_tet_set_np = np.array(meshData['sum_tet_set'], dtype=int)

    # 是否使用各向异性
    # dyn_model = dyn.Dynamics_XPBD_SNH_Active_aniso(body=geo_model, num_pts_np=num_per_tet_set_np, tag_dirichlet_all_dir=flag_dirichlet, tag_neumann=flag_neumann)
    dyn_model = dyn.Dynamics_XPBD_SNH_Active(body=geo_model, num_pts_np=num_per_tet_set_np, tag_dirichlet_all_dir=flag_dirichlet, tag_neumann=flag_neumann)
    save_path = args.save_path

    mygui = gui.Gui(geometry_model=geo_model, dynamics_model=dyn_model, body_name=body_name, save_path=save_path)
    mygui.interaction_operator.open_interaction_during_solving = args.dyn_ix  # 动态交互  

    # 场景配置
    mygui.set_electrophysiology_model_type("ap")
    mygui.interaction_operator.dyn_op.Youngs_modulus = 1000.0
    mygui.interaction_operator.dyn_op.Poisson_ratio = 0.49

    # 开启gui渲染
    mygui.display()

def set_scene_biventricular_em_scroll_pulse(args):

    # 导入数据
    from project.data.biventricular import meshData
    body_name = meshData['name']
    geo_model, flag_dirichlet, flag_neumann = geo.read_body(meshData=meshData)
    num_per_tet_set_np = np.array(meshData['sum_tet_set'], dtype=int)

    # 是否使用各向异性
    # dyn_model = dyn.Dynamics_XPBD_SNH_Active_aniso(body=geo_model, num_pts_np=num_per_tet_set_np, tag_dirichlet_all_dir=flag_dirichlet, tag_neumann=flag_neumann)
    dyn_model = dyn.Dynamics_XPBD_SNH_Active(body=geo_model, num_pts_np=num_per_tet_set_np, tag_dirichlet_all_dir=flag_dirichlet, tag_neumann=flag_neumann)
    save_path = args.save_path

    mygui = gui.Gui(geometry_model=geo_model, dynamics_model=dyn_model, body_name=body_name, save_path=save_path)
    mygui.interaction_operator.open_interaction_during_solving = args.dyn_ix  # 动态交互  

    # 场景配置
    mygui.set_electrophysiology_model_type("ap")
    mygui.interaction_operator.dyn_op.Youngs_modulus = 1000.0
    mygui.interaction_operator.dyn_op.Poisson_ratio = 0.49
    dyn_model.pressure = -50.0

    # 开启gui渲染
    mygui.display()