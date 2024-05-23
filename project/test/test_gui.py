import taichi as ti
import numpy as np
import project.Geometry as geo
import project.Electrophysiology as elec
import project.Dynamics as dyn
import project.GUI as gui

from project.data.whole_heart import meshData


if __name__ == "__main__":
    ti.init(arch=ti.gpu, default_fp=ti.float32)

    body_name = meshData['name']
    geo_model, flag_dirichlet, flag_neumann = geo.read_body(meshData=meshData)
    num_per_tet_set_np = np.array(meshData['sum_tet_set'], dtype=int)
    dyn_model = dyn.Dynamics_XPBD_SNH_Active_aniso(body=geo_model, num_pts_np=num_per_tet_set_np)

    mygui = gui.Gui(geometry_model=geo_model, dynamics_model=dyn_model, body_name=body_name)

    mygui.display()
