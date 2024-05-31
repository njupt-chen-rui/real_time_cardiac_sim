import taichi as ti
import numpy as np
import project.Configure as cfg
import project.Geometry as geo
import project.Electrophysiology as elec
import project.Dynamics as dyn
import project.GUI as gui
import argparse
import os
import project.PreScene as psc


def set_usr_scene(args):
    str = args.body_name
    if str == "whole_heart":
        from project.data.whole_heart import meshData
    elif str == "cube":
        from project.data.cube import meshData
    elif str == "biventricular":
        from project.data.biventricular import meshData
    
    body_name = meshData['name']
    geo_model, flag_dirichlet, flag_neumann = geo.read_body(meshData=meshData)
    num_per_tet_set_np = np.array(meshData['sum_tet_set'], dtype=int)
    dyn_model = dyn.Dynamics_XPBD_SNH_Active_aniso(body=geo_model, num_pts_np=num_per_tet_set_np)
    save_path = args.save_path
    mygui = gui.Gui(geometry_model=geo_model, dynamics_model=dyn_model, body_name=body_name, save_path=save_path)
    mygui.interaction_operator.open_interaction_during_solving = args.dyn_ix

    mygui.display()


# TODO:
def set_scene_whole_heart(args):
    from project.data.whole_heart import meshData
    body_name = meshData['name']
    geo_model, flag_dirichlet, flag_neumann = geo.read_body(meshData=meshData)
    num_per_tet_set_np = np.array(meshData['sum_tet_set'], dtype=int)
    dyn_model = dyn.Dynamics_XPBD_SNH_Active_aniso(body=geo_model, num_pts_np=num_per_tet_set_np)
    save_path = args.save_path
    mygui = gui.Gui(geometry_model=geo_model, dynamics_model=dyn_model, body_name=body_name, save_path=save_path)
    mygui.interaction_operator.open_interaction_during_solving = args.dyn_ix

    mygui.display()


# def set_scene_cube(args):
#     from project.data.cube import meshData
#     body_name = meshData['name']
#     geo_model, flag_dirichlet, flag_neumann = geo.read_body(meshData=meshData)
#     num_per_tet_set_np = np.array(meshData['sum_tet_set'], dtype=int)
#     dyn_model = dyn.Dynamics_XPBD_SNH_Active_aniso(body=geo_model, num_pts_np=num_per_tet_set_np, tag_dirichlet_all_dir=flag_dirichlet, tag_neumann=flag_neumann)
#     save_path = args.save_path
#     mygui = gui.Gui(geometry_model=geo_model, dynamics_model=dyn_model, body_name=body_name, save_path=save_path)

#     mygui.display()


def set_scene(args):
    cfg.Preset_Scene = args.scene
    scene_id = args.scene
    cfg
    if scene_id == 0:
        set_usr_scene(args)
    elif scene_id == 1:
        set_scene_whole_heart(args)
    elif scene_id == 3:
        # set_scene_cube(args)
        psc.set_scene_cube(args)


if __name__ == "__main__":
    ti.init(arch=ti.gpu, default_fp=ti.float32)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default='./res/')
    parser.add_argument('--scene', type=int, default=0)
    parser.add_argument('--body_name', type=str, default='whole_heart')
    parser.add_argument('--dyn_ix', type=bool, default=False, help='when it is True, open Dynamic interaction')
    args = parser.parse_args()

    assert 0 <= args.scene <= 5

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    set_scene(args)
