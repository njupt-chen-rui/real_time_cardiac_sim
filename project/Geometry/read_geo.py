import numpy as np
import project.Geometry as geo
import project.tool as tool


# TODO: 把读文件写成一个类(要读的属性有些多)
""" 
 - 改成使用路径读取文件
 - 生成body
 - 返回flag_dirichlet 和 flag_neumann
 - 读取num_tet_set
 - 读取 meshData 的 name
"""
def read_body(meshData):
    """从文件中读入数据并创建Body对象

    :return: Body对象, flag_dirichlet, flag_neumann
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
    flag_dirichlet = False
    if 'bou_tag_dirichlet' in meshData:
        bou_tag_dirichlet_np = np.array(meshData['bou_tag_dirichlet'], dtype=int)
        flag_dirichlet = True
    elif 'bou_base_face' in meshData:
        bou_tag_dirichlet_face_np = np.array(meshData['bou_base_face'], dtype=int)
        bou_tag_dirichlet_np = np.zeros(len(pos_np), dtype=int)
        get_bou_tag_dirichlet_from_face(bou_tag_dirichlet_np, bou_tag_dirichlet_face_np)
        flag_dirichlet = True
    else:
        bou_tag_dirichlet_np = np.zeros(len(pos_np), dtype=int)

    # neumann bou
    flag_neumann = False
    if ('bou_endo_lv_face' in meshData) and ('bou_endo_rv_face' in meshData):
        bou_endo_lv_face_np = np.array(meshData['bou_endo_lv_face'], dtype=int)
        bou_endo_lv_face_np = bou_endo_lv_face_np.reshape((-1, 3))
        bou_endo_rv_face_np = np.array(meshData['bou_endo_rv_face'], dtype=int)
        bou_endo_rv_face_np = bou_endo_rv_face_np.reshape((-1, 3))
        bou_neumann_face_np = np.append(bou_endo_lv_face_np, bou_endo_rv_face_np, axis=0)
        flag_neumann = True
    elif 'bou_endo_lv_face' in meshData:
        bou_endo_lv_face_np = np.array(meshData['bou_endo_lv_face'], dtype=int)
        bou_endo_lv_face_np = bou_endo_lv_face_np.reshape((-1, 3))
        bou_neumann_face_np = bou_endo_lv_face_np
        flag_neumann = True
    # elif 'bou_tag_neumann' in meshData:
    #     bou_tag_neumann_np = np.array(meshData['bou_tag_neumann'], dtype=int)
    #     flag_neumann = True
    else:
        bou_neumann_face_np = np.zeros(3, dtype=int)
        bou_neumann_face_np = bou_neumann_face_np.reshape((-1, 3))
        

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
                 bou_tag_neumann_np=bou_neumann_face_np
                 )

    return body, flag_dirichlet, flag_neumann

def get_bou_tag_dirichlet_from_face(node_np, face_np):
    for i in range(len(face_np)):
        vid = face_np[i]
        node_np[vid] = 1
