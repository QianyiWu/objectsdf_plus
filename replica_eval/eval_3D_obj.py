import numpy as np
from tqdm import tqdm
import trimesh
from metrics import accuracy, completion, completion_ratio
import os
import json
import glob

def calc_3d_metric(mesh_rec, mesh_gt, N=200000):
    """
    3D reconstruction metric.
    """
    metrics = [[] for _ in range(8)]
    transform, extents = trimesh.bounds.oriented_bounds(mesh_gt)
    extents = extents / 0.9  if N != 200000 else extents # enlarge 0.9 for objects
    # extents = extents *1.2  if N != 200000 else extents # enlarge 0.9 for objects
    box = trimesh.creation.box(extents=extents, transform=np.linalg.inv(transform))
    mesh_rec = mesh_rec.slice_plane(box.facets_origin, -box.facets_normal)
    if mesh_rec.vertices.shape[0] == 0:
        print("no mesh found")
        return
    rec_pc = trimesh.sample.sample_surface(mesh_rec, N)
    rec_pc_tri = trimesh.PointCloud(vertices=rec_pc[0])

    gt_pc = trimesh.sample.sample_surface(mesh_gt, N)
    gt_pc_tri = trimesh.PointCloud(vertices=gt_pc[0])
    accuracy_rec = accuracy(gt_pc_tri.vertices, rec_pc_tri.vertices)
    completion_rec = completion(gt_pc_tri.vertices, rec_pc_tri.vertices)
    completion_ratio_rec = completion_ratio(gt_pc_tri.vertices, rec_pc_tri.vertices, 0.05)
    completion_ratio_rec_1 = completion_ratio(gt_pc_tri.vertices, rec_pc_tri.vertices, 0.01)

    precision_ratio_rec = completion_ratio(rec_pc_tri.vertices, gt_pc_tri.vertices, 0.05)
    precision_ratio_rec_1 = completion_ratio(rec_pc_tri.vertices, gt_pc_tri.vertices, 0.01)

    f_score = 2*precision_ratio_rec*completion_ratio_rec / (completion_ratio_rec + precision_ratio_rec)
    f_score_1 = 2 * precision_ratio_rec_1*completion_ratio_rec_1 / (completion_ratio_rec_1 + precision_ratio_rec_1)


    # accuracy_rec *= 100  # convert to cm
    # completion_rec *= 100  # convert to cm
    # completion_ratio_rec *= 100  # convert to %
    # print('accuracy: ', accuracy_rec)
    # print('completion: ', completion_rec)
    # print('completion ratio: ', completion_ratio_rec)
    # print("completion_ratio_rec_1cm ", completion_ratio_rec_1)
    metrics[0].append(accuracy_rec)
    metrics[1].append(completion_rec)
    metrics[2].append(completion_ratio_rec_1)
    metrics[3].append(completion_ratio_rec)
    metrics[4].append(precision_ratio_rec_1)
    metrics[5].append(precision_ratio_rec)
    metrics[6].append(np.nan_to_num(f_score_1))
    metrics[7].append(np.nan_to_num(f_score))

    return metrics

def get_gt_bg_mesh(gt_dir, background_cls_list):
    with open(os.path.join(gt_dir, "info_semantic.json")) as f:
        label_obj_list = json.load(f)["objects"]

    bg_meshes = []
    for obj in label_obj_list:
        if int(obj["class_id"]) in background_cls_list:
            obj_file = os.path.join(gt_dir, "mesh_semantic.ply_" + str(int(obj["id"])) + ".ply")
            obj_mesh = trimesh.load(obj_file)
            bg_meshes.append(obj_mesh)

    bg_mesh = trimesh.util.concatenate(bg_meshes)
    return bg_mesh

def get_obj_ids(obj_dir):
    files = os.listdir(obj_dir)
    obj_ids = []
    for f in files:
        obj_id = f.split("obj")[1][:-1]
        if obj_id == '':
            continue
        obj_ids.append(int(obj_id))
    return obj_ids

def get_obj_ids_ours(obj_dir):
    files = list(filter(os.path.isfile, glob.glob(os.path.join(obj_dir, '*[0-9].ply'))))
    epoch_count = set([int(os.path.basename(f).split('_')[1]) for f in files])
    max_epoch = max(epoch_count)
    # print(max_epoch)
    obj_file = [int(os.path.basename(f).split('.')[0].split('_')[2]) for f in files if int(os.path.basename(f).split('_')[1])==max_epoch]
    return sorted(obj_file), max_epoch



def get_gt_mesh_from_objid(gt_dir, mapping_list):
    combined_meshes = []
    if len(mapping_list) == 1:
        return trimesh.load(os.path.join(gt_dir, 'cull_mesh_semantic.ply_'+str(mapping_list[0])+'.ply'))
    else:
        for idx in mapping_list:
            if os.path.isfile(os.path.join(gt_dir, 'cull_mesh_semantic.ply_'+str(int(idx))+'.ply')):
                obj_file = os.path.join(gt_dir, 'cull_mesh_semantic.ply_'+str(int(idx))+'.ply')
                obj_mesh = trimesh.load(obj_file)
                combined_meshes.append(obj_mesh)
            else:
                continue
        combine_mesh = trimesh.util.concatenate(combined_meshes)
        return combine_mesh



if __name__ == "__main__":
    exp_name = ["room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4"]
    data_dir = "/media/hdd/Replica-Dataset/vmap" # where to store the data
    log_dir = './evaluation_results' # where to store the evaluation results
    info_dir = '../data/replica' # path to dataset information
    mesh_rec_root = "../exps/objectsdfplus_replica" # path to the reconstruction results
    os.makedirs(log_dir, exist_ok=True)
    for idx, exp in enumerate(exp_name):
        if exp is not "room2": continue
        idx = idx + 1
        gt_dir = os.path.join(data_dir, exp[:-1]+"_"+exp[-1]+"/habitat")
        info_text = os.path.join(info_dir, 'scan'+str(idx), 'instance_mapping.txt')
        # mesh_dir = os.path.join()
        # mesh_rec_dir = os.path.join('') # path to reconstruction results
        # get the lastest folder for evaluation
        dirs = sorted(os.listdir(mesh_rec_root+f'_{idx}'))
        mesh_rec_dir = os.path.join(mesh_rec_root+f'_{idx}', dirs[-1], "plots")
        print(mesh_rec_dir)

        output_dir = os.path.join(log_dir, exp+'_{}'.format(mesh_rec_root.split('/')[-1]))
        os.makedirs(output_dir, exist_ok=True)
        metrics_3D = [[] for _ in range(8)]

        #  only calculate the valid mesh in experiment
        instance_mapping = {}
        with open(info_text, 'r') as f:
            for l in f:
                (k, v_sem, v_ins) = l.split(',')
                instance_mapping[int(k)] = int(v_ins)
        label_mapping = sorted(set(instance_mapping.values()))
        # print(label_mapping)
        # get all valid obj index
        obj_ids, max_epoch = get_obj_ids_ours(mesh_rec_dir)
        # print(obj_ids)
        for obj_id in tqdm(obj_ids):
            inst_id = label_mapping[obj_id]
            # merge the gt mesh with the same instance_id
            gt_inst_list = [] # a list used to store the index in gt_object mesh that are the same object defined by instance_mapping
            for k, v in instance_mapping.items():
                if v == inst_id:
                    gt_inst_list.append(k)
            mesh_gt = get_gt_mesh_from_objid(os.path.join(gt_dir, 'cull_object_mesh'), gt_inst_list)
            if obj_id == 0:
                N=200000
            else:
                N=10000

            # in order to evaluate the result, we need to cull the object mesh first and then evaluate the metric
            # mesh_rec = trimesh.load(os.path.join(mesh_dir, 'surface_{max_epoch}_{obj_id}.ply'))
            cull_rec_mesh_path = os.path.join(output_dir, f"{exp}_cull_surface_{max_epoch}_{obj_id}.ply")

            rec_mesh = os.path.join(mesh_rec_dir, f'surface_{max_epoch}_{obj_id}.ply')
            # print(rec_mesh)
            cmd = f"python cull_mesh.py --input_mesh {rec_mesh} --input_scalemat ../data/replica/scan{idx}/cameras.npz --traj ../data/replica/scan{idx}/traj.txt --output_mesh {cull_rec_mesh_path}"
            os.system(cmd)
            # evaluate the metric 
            mesh_rec = trimesh.load(cull_rec_mesh_path)
            # use the biggest connected component for evaluation.

            metrics = calc_3d_metric(mesh_rec, mesh_gt, N=N)
            if metrics is None:
                continue
            np.save(output_dir + '/metric_obj{}.npy'.format(obj_id), np.array(metrics))
            metrics_3D[0].append(metrics[0])    # acc
            metrics_3D[1].append(metrics[1])    # comp
            metrics_3D[2].append(metrics[2])    # comp ratio 1cm
            metrics_3D[3].append(metrics[3])    # comp ratio 5cm
            metrics_3D[4].append(metrics[4])    # precision ratio 1cm
            metrics_3D[5].append(metrics[5])    # precision ration 5cm
            metrics_3D[6].append(metrics[6])    # f_score 1
            metrics_3D[7].append(metrics[7])    # f_score 5cm
        metrics_3D = np.array(metrics_3D)
        np.save(output_dir + '/metrics_3D_obj.npy', metrics_3D)
        print("metrics 3D obj \n Acc | Comp | Comp Ratio 1cm | Comp Ratio 5cm \n", metrics_3D.mean(axis=1))
        print("-----------------------------------------")
        print("finish exp ", exp)
        # exit()



        
        # use culled object mesh for evaluation
    

        



    # background_cls_list = [5, 12, 30, 31, 40, 60, 92, 93, 95, 97, 98, 79]
    # exp_name = ["room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4"]
    # # data_dir = "/home/xin/data/vmap/"
    # data_dir = "/media/hdd/Replica-Dataset/vmap/"
    # # log_dir = "../logs/iMAP/"
    # # log_dir = "../logs/vMAP/"
    # log_dir = "/media/hdd/Replica-Dataset/vmap/vMAP_Replica_Results/"

    # for exp in tqdm(exp_name):
    #     gt_dir = os.path.join(data_dir, exp[:-1]+"_"+exp[-1]+"/habitat")
    #     exp_dir = os.path.join(log_dir, exp+'_vmap')
    #     mesh_dir = os.path.join(exp_dir, "scene_mesh")
    #     output_path = os.path.join(exp_dir, "eval_mesh")
    #     os.makedirs(output_path, exist_ok=True)
    #     metrics_3D = [[] for _ in range(4)]

    #     # get obj ids
    #     # obj_ids = np.loadtxt()    # todo use a pre-defined obj list or use vMAP results
    #     obj_ids = get_obj_ids(mesh_dir.replace("imap", "vmap"))
    #     for obj_id in tqdm(obj_ids):
    #         if obj_id == 0: # for bg
    #             N = 200000
    #             mesh_gt = get_gt_bg_mesh(gt_dir, background_cls_list)
    #         else:   # for obj
    #             N = 10000
    #             obj_file = os.path.join(gt_dir, "mesh_semantic.ply_" + str(obj_id) + ".ply")
    #             mesh_gt = trimesh.load(obj_file)

    #         if "vMAP" in exp_dir:
    #             rec_meshfile = os.path.join(mesh_dir, "imap_frame2000_obj"+str(obj_id)+".obj")
    #             # rec_meshfile = os.path.join(mesh_dir, )
    #         elif "iMAP" in exp_dir:
    #             rec_meshfile = os.path.join(mesh_dir, "frame_1999_obj0.obj")
    #         else:
    #             print("Not Implement")
    #             exit(-1)

    #         mesh_rec = trimesh.load(rec_meshfile)
    #         # mesh_rec.invert()   # niceslam mesh face needs invert
    #         metrics = calc_3d_metric(mesh_rec, mesh_gt, N=N)  # for objs use 10k, for scene use 200k points
    #         if metrics is None:
    #             continue
    #         np.save(output_path + '/metric_obj{}.npy'.format(obj_id), np.array(metrics))
    #         metrics_3D[0].append(metrics[0])    # acc
    #         metrics_3D[1].append(metrics[1])    # comp
    #         metrics_3D[2].append(metrics[2])    # comp ratio 1cm
    #         metrics_3D[3].append(metrics[3])    # comp ratio 5cm
    #     metrics_3D = np.array(metrics_3D)
    #     np.save(output_path + '/metrics_3D_obj.npy', metrics_3D)
    #     print("metrics 3D obj \n Acc | Comp | Comp Ratio 1cm | Comp Ratio 5cm \n", metrics_3D.mean(axis=1))
    #     print("-----------------------------------------")
    #     print("finish exp ", exp)
    
    # calculate the avaerage result over all 8 scenes
