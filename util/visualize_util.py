import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

def Visualize(V, F, L, L_gt):
	face_labels = L
	gt_face_labels = L_gt
	num_labels = np.max(face_labels) + 1
	colors = np.random.rand(num_labels, 3)
	fp = open('visual/results/visualize/face.obj', 'w')
	voffset = 1
	for i in range(F.shape[0]):
		if face_labels[i] < 0:
			c = np.array([0,0,0])
		else:
			c = colors[face_labels[i]]
		for j in range(3):
			v = V[F[i][j]]
			fp.write('v %f %f %f %f %f %f\n'%(v[0], v[1], v[2], c[0], c[1], c[2]))
		fp.write('f %d %d %d\n'%(voffset, voffset + 1, voffset + 2))
		voffset += 3
	fp.close()

	num_labels = np.max(gt_face_labels) + 1
	colors = np.random.rand(num_labels, 3)
	fp = open('visual/results/visualize/face_gt.obj', 'w')
	voffset = 1
	for i in range(F.shape[0]):
		if gt_face_labels[i] < 0:
			c = np.array([0,0,0])
		else:
			c = colors[gt_face_labels[i]]
		for j in range(3):
			v = V[F[i][j]]
			fp.write('v %f %f %f %f %f %f\n'%(v[0], v[1], v[2], c[0], c[1], c[2]))
		fp.write('f %d %d %d\n'%(voffset, voffset + 1, voffset + 2))
		voffset += 3
	fp.close()
	
def visualize_prim_face(fn, V, F, L, L_gt):
    colors = np.random.rand(10000, 3)
    VC = (V[F[:,0]] + V[F[:,1]] + V[F[:,2]]) / 3.0

    fp = open('visual/results/visualize/%s.obj'%(fn[0][:8]+'ours'), 'w')
    for i in range(VC.shape[0]):
        v = VC[i]
        if L[i] < 0:
            p = np.array([0,0,0])
        else:
            p = colors[L[i]]
        fp.write('v %f %f %f %f %f %f\n'%(v[0],v[1],v[2],p[0],p[1],p[2]))
    fp.close()

    # fp = open('visual/results/visualize-gt/%s.obj'%(fn[0]), 'w')
    # for i in range(VC.shape[0]):
    #     v = VC[i]
    #     if L_gt[i] < 0:
    #         p = np.array([0,0,0])
    #     else:
    #         p = colors[L_gt[i]]
    #     fp.write('v %f %f %f %f %f %f\n'%(v[0],v[1],v[2],p[0],p[1],p[2]))
    # fp.close()

def visualize_prim_point(fn, V, L):
    colors = np.random.rand(10000, 3)
    # VC = (V[F[:,0]] + V[F[:,1]] + V[F[:,2]]) / 3.0

    fp = open('results/ours/visualize/%s.obj'%(fn[0][:8]+'ours'), 'w')
    for i in range(V.shape[0]):
        v = V[i]
        if L[i] < 0:
            p = np.array([0,0,0])
        else:
            p = colors[L[i]]
        fp.write('v %f %f %f %f %f %f\n'%(v[0],v[1],v[2],p[0],p[1],p[2]))
    fp.close()

def open3d_vis_prim_face(fn, V, F, L, L_gt):
    
    VC = (V[F[:,0]] + V[F[:,1]] + V[F[:,2]]) / 3.0
    colors = np.random.rand(len(VC), 3)

    back_point_mask = L < 0
    back_piont = VC[back_point_mask]
    point_idx = np.where(L >= 0)
    point = VC[~back_point_mask]

    tree = cKDTree(point)
    _, idx = tree.query(back_piont, k=1)
    L[back_point_mask] = L[point_idx[0][idx]]
    for i in range(VC.shape[0]):
        if L[i] < 0:
            import ipdb
            ipdb.set_trace()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(VC)
    pcd.colors = o3d.utility.Vector3dVector(colors[L])
    o3d.io.write_point_cloud('results/ours/visualize/%s.ply'%(fn[0][:8]+'ours'), pcd)

    # back_point_mask = L_gt < 0
    # back_piont = VC[back_point_mask]
    # point_idx = np.where(L_gt >= 0)
    # point = VC[~back_point_mask]

    # tree = cKDTree(point)
    # _, idx = tree.query(back_piont, k=1)
    # L_gt[back_point_mask] = L_gt[point_idx[0][idx]]
    # for i in range(VC.shape[0]):
    #     if L_gt[i] < 0:
    #         import ipdb
    #         ipdb.set_trace()
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(VC)
    # pcd.colors = o3d.utility.Vector3dVector(colors[L_gt])
    # o3d.io.write_point_cloud('visual/gt/%s.ply'%(fn[0][:8]), pcd)