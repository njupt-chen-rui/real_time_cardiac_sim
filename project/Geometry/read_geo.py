import numpy as np
import project.Geometry as geo
import project.tool as tool


def read_body(meshData):
    """从文件中读入数据并创建Body对象

    :return: Body对象
    """

    # 顶点位置
    pos_np = np.array(meshData['verts'], dtype=float)
    pos_np = pos_np.reshape((-1, 3))
    # 四面体顶点索引
    tet_np = np.array(meshData['tetIds'], dtype=int)
    tet_np = tet_np.reshape((-1, 4))
    # tet_fiber方向
    fiber_tet_np = np.array(meshData['fiberDirection'], dtype=float)
    fiber_tet_np = fiber_tet_np.reshape((-1, 3))
    # tet_sheet方向
    sheet_tet_np = np.array(meshData['sheetDirection'], dtype=float)
    sheet_tet_np = sheet_tet_np.reshape((-1, 3))
    # tet_normal方向
    normal_tet_np = np.array(meshData['normalDirection'], dtype=float)
    normal_tet_np = normal_tet_np.reshape((-1, 3))
    # num_tet_set
    num_tet_set_np = np.array(meshData['num_tet_set'], dtype=int)[0]
    # tet_set
    tet_set_np = np.array(meshData['tet_set'], dtype=int)
    # dirichlet bou
    if meshData['bou_tag_dirichlet']:
        bou_tag_dirichlet_np = np.array(meshData['bou_tag_dirichlet'], dtype=int)
    else:
        bou_tag_dirichlet_np = np.zeros(len(pos_np), dtype=int)

    # neumann bou
    if meshData['bou_tag_neumann']:
        bou_tag_neumann_np = np.array(meshData['bou_tag_neumann'], dtype=int)
    else:
        bou_tag_neumann_np = np.zeros(len(pos_np), dtype=int)

    # colormap
    colormap = tool.Colormap()

    body = geo.Body(
                 colormap=colormap,
                 nodes_np=pos_np,
                 elements_np=tet_np,
                 tet_fiber_np=fiber_tet_np,
                 tet_sheet_np=sheet_tet_np,
                 tet_normal_np=normal_tet_np,
                 num_tet_set_np=num_tet_set_np,
                 tet_set_np=tet_set_np,
                 bou_tag_dirichlet_np=bou_tag_dirichlet_np,
                 bou_tag_neumann_np=bou_tag_neumann_np
                 )

    return body
