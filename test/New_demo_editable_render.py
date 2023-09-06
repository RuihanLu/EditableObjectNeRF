import sys
import os

os.environ["OMP_NUM_THREADS"] = "1"  # noqa
os.environ["MKL_NUM_THREADS"] = "1"  # noqa
sys.path.append(".")  # noqa

import imageio, torch
import numpy as np
from tqdm import tqdm
from render_tools.editable_renderer import EditableRenderer, read_testing_config
from utils.util import get_timestamp
from scipy.spatial.transform import Rotation
from render_tools.multi_rendering import volume_rendering_multi
from collections import defaultdict

def move_camera_pose(pose, progress):
    # control the camera move (spiral pose)
    t = progress * np.pi * 4
    radii = 0.01
    center = np.array([np.cos(t), -np.sin(t), -np.sin(0.5 * t)]) * radii
    pose[:3, 3] += pose[:3, :3] @ center
    return pose

def get_pure_rotation(progress_11: float, max_angle: float = 180):
    trans_pose = np.eye(4)
    trans_pose[:3, :3] = Rotation.from_euler(
        "z", progress_11 * max_angle, degrees=True
    ).as_matrix()
    return trans_pose

def get_transformation_with_duplication_offset(progress, duplication_id: int):
    trans_pose = get_pure_rotation(np.sin(progress * np.pi * 2), max_angle=10)
    offset = 0.05
    if duplication_id > 0:
        trans_pose[0, 3] -= np.sin(progress * np.pi * 2) * offset
        trans_pose[1, 3] -= 0.2
    else:
        trans_pose[0, 3] += np.sin(progress * np.pi * 2) * offset
        trans_pose[1, 3] += 0.55
    return trans_pose

def GenerateRender(config):
    renderer = EditableRenderer(config=config)
    renderer.load_frame_meta()
    obj_id_list = config.obj_id_list  # e.g. [4, 6]
    for obj_id in obj_id_list:
        renderer.initialize_object_bbox(obj_id)
    renderer.remove_scene_object_by_ids(obj_id_list)
    return renderer

def main(back_config, obj_config, objScale=1, xMove=0, yMove=0, zMove=0):
    # render_path = f"debug/rendered_view/render_{get_timestamp()}_{back_config.prefix}_{obj_config.prefix}/"
    render_path = f"debug/rendered_view/render__{back_config.prefix}_{obj_config.prefix}/"
    print('SavePath: ', render_path)
    os.makedirs(render_path, exist_ok=True)

    back_renderer = GenerateRender(back_config)
    back_obj_id_list = back_config.obj_id_list
    obj_renderer = GenerateRender(obj_config)
    obj_obj_id_list = obj_config.obj_id_list

    W, H = back_config.img_wh
    back_total_frames = back_config.total_frames
    back_pose_frame_idx = back_config.test_frame

    obj_total_frames = obj_config.total_frames
    obj_pose_frame_idx = obj_config.test_frame

    for idx in tqdm(range(min(back_total_frames, obj_total_frames))):
        processed_obj_id = []
        for obj_id in back_obj_id_list:
            # count object duplication, which is generally to be zero,
            # but can be increased if duplication operation happened
            obj_duplication_cnt = np.sum(np.array(processed_obj_id) == obj_id)
            progress = idx / back_total_frames

            if back_config.edit_type == "duplication":
                trans_pose = get_transformation_with_duplication_offset(
                    progress, obj_duplication_cnt
                )
            elif back_config.edit_type == "pure_rotation":
                trans_pose = get_pure_rotation(progress_11=(progress * 2 - 1))
            back_renderer.set_object_pose_transform(obj_id, trans_pose, obj_duplication_cnt)
            processed_obj_id.append(obj_id)
        processed_obj_id = []
        for obj_id in obj_obj_id_list:
            # count object duplication, which is generally to be zero,
            # but can be increased if duplication operation happened
            obj_duplication_cnt = np.sum(np.array(processed_obj_id) == obj_id)
            progress = idx / obj_total_frames

            if obj_config.edit_type == "duplication":
                trans_pose = get_transformation_with_duplication_offset(
                    progress, obj_duplication_cnt
                )
            elif obj_config.edit_type == "pure_rotation":
                trans_pose = get_pure_rotation(progress_11=(progress * 2 - 1))
            obj_renderer.set_object_pose_transform(obj_id, trans_pose, obj_duplication_cnt)
            processed_obj_id.append(obj_id)

        backCameraPos = move_camera_pose(
                back_renderer.get_camera_pose_by_frame_idx(back_pose_frame_idx),
                idx / back_total_frames,
            )
        objCameraPos = move_camera_pose(
            obj_renderer.get_camera_pose_by_frame_idx(obj_pose_frame_idx),
            idx / obj_total_frames,
        )
        # render edited scene
        back_rendered_ray_chunks_ls = back_renderer.render_edit2(
            h=H,
            w=W,
            camera_pose_Twc=backCameraPos,
            fovx_deg=getattr(back_renderer, "fov_x_deg_dataset", 60),
            # render_obj_only=True
        )
        obj_rendered_ray_chunks_ls = obj_renderer.render_edit2(
            h=H,
            w=W,
            camera_pose_Twc=objCameraPos,
            fovx_deg=getattr(obj_renderer, "fov_x_deg_dataset", 60),
            render_obj_only=True,
            scale=objScale,
            xMove=xMove,
            yMove=yMove,
            zMove=zMove,
        )

        results = defaultdict(list)
        for back_info, obj_info in zip(back_rendered_ray_chunks_ls, obj_rendered_ray_chunks_ls):
            tRes = back_info[0]
            z_vals_fine_list = back_info[1] + obj_info[1]
            rgbs_list = back_info[2] + obj_info[2]
            sigmas_list = back_info[3] + obj_info[3]
            noise_std = (back_info[4] + obj_info[4]) / 2
            white_back = (back_info[5] + obj_info[5]) / 2
            volume_rendering_multi(
                tRes,
                "fine",
                z_vals_fine_list,
                rgbs_list,
                sigmas_list,
                noise_std,
                white_back,
            )
            for k, v in tRes.items():
                results[k] += [v.detach().cpu()]
        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        image_out_path = f"{render_path}/render_{idx:04d}.png"
        image_np = results["rgb_fine"].view(H, W, 3).detach().cpu().numpy()
        imageio.imwrite(image_out_path, (image_np * 255).astype(np.uint8))

        back_renderer.reset_active_object_ids()
        obj_renderer.reset_active_object_ids()
        # return

if __name__ == "__main__":
    # Scene
    back_config = r'test/config/edit_scannet_0113.yaml'
    back_ckpt_path = r'./ckptLs/scannet_0113/scannet0113.ckpt'
    back_prefix = r'scannet_0113_duplicating_moving'
    # Object
    obj_config = r'test/config/edit_toy_desk_2.yaml'
    obj_ckpt_path = r'./ckptLs/toydesk2/toydesk2.ckpt'
    obj_prefix = r'toydesk2_duplicating_moving'
    #Obj scaling and movement parameters, where xMove, yMove, zMove parameter values can be referred to the pixel coordinates, and pixel coordinates are similar, but not a one-to-one correspondence, subject to scaling and projection principle influence
    objScale = 0.8      # obj zoom ratio
    xMove = -10         # Image coordinate system x-direction translation, moving to the right is positive, moving to the left is negative
    yMove = 40          # The image coordinate system is translated in the y-direction, i.e., up is positive, down is negative
    zMove = -40         # Image coordinate system translation, positively close to the obj, the obj will become larger, negatively far from the obj, the obj will become smaller

    back_config = read_testing_config(back_config, back_ckpt_path, back_prefix)
    obj_config = read_testing_config(obj_config, obj_ckpt_path, obj_prefix)
    main(back_config, obj_config, objScale=objScale, xMove=xMove, yMove=yMove, zMove=zMove)

'''
config=test/config/edit_scannet_0113.yaml ckpt_path=./ckptLs/toydesk2/toydesk2.ckpt prefix=scannet_0113_duplicating_moving

python test/demo_editable_render.py config=test/config/edit_scannet_0113.yaml ckpt_path=./ckptLs/scannet_0113/scannet0113.ckpt prefix=scannet_0113_duplicating_moving
python test/demo_editable_render.py config=test/config/edit_toy_desk_2.yaml ckpt_path=./ckptLs/toydesk2/toydesk2.ckpt prefix=toydesk2_duplicating_moving

python test/demo_editable_render.py \
    config=test/config/edit_toy_desk_2.yaml \
    ckpt_path=../object_nerf_edit_demo_models/toydesk_2/last.ckpt \
    prefix=toy_desk2_rotating
'''
