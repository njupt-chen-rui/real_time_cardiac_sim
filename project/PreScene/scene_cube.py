import taichi as ti
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

    set_dirichlet_bou(geo_model=geo_model)
    mygui = gui.Gui(geometry_model=geo_model, dynamics_model=dyn_model, body_name=body_name, save_path=save_path)

    mygui.display()


@ti.kernel
def set_dirichlet_bou(geo_model: ti.template()):
    for i in geo_model.nodes:
        if abs(geo_model.nodes[i][1]) < 1e-12:
            geo_model.bou_tag_dirichlet[i] = 1
