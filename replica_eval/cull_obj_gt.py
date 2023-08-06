import os
import glob


scans = ["room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4"]

gt_data_dir = '/media/hdd/Replica-Dataset/vmap/'

for idx, exp in enumerate(scans):
    idx = idx + 1
    folder_name = os.path.join(gt_data_dir, exp[:-1]+'_'+exp[-1], 'habitat')
    files = list(filter(os.path.isfile, glob.glob(os.path.join(folder_name, 'mesh_semantic.*.ply'))))
    # print(files)
    # exit()
    print(files[0].split('/')[-1])
    os.makedirs(os.path.join(folder_name, 'cull_object_mesh'), exist_ok=True)
    for name in files:
        cull_mesh_out = os.path.join(folder_name, 'cull_object_mesh', 'cull_'+name.split('/')[-1])
        # print(cull_mesh_out)
        cmd = f"python cull_mesh.py --input_mesh {name} --traj ../data/replica/Replica/scan{idx}/traj.txt --output_mesh {cull_mesh_out}"
        print(cmd)
        os.system(cmd)
    # exit()
