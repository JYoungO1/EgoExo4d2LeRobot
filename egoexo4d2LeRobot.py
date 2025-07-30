import argparse
import os
import shutil
import json
import imageio
import numpy as np
from pathlib import Path
import time
import pandas as pd
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import (
    write_episode,
    write_episode_stats,
    write_info,
    write_task,
)

def extract_observation_state_from_3d(annotation3D, joints_order):
    state = []
    for joint in joints_order:
        if joint in annotation3D:
            joint_data = annotation3D[joint]
            state.extend([joint_data["x"], joint_data["y"], joint_data["z"]])
        else:
            state.extend([0.0, 0.0, 0.0])
    return np.array(state, dtype=np.float32)

joints_order = [
    "right_wrist",
    "right_thumb_1", "right_thumb_2", "right_thumb_3", "right_thumb_4",
    "right_index_1", "right_index_2", "right_index_3", "right_index_4",
    "right_middle_1", "right_middle_2", "right_middle_3", "right_middle_4",
    "right_ring_1", "right_ring_2", "right_ring_3", "right_ring_4",
    "right_pinky_1", "right_pinky_2", "right_pinky_3", "right_pinky_4",
]

def validate_all_metadata(all_metadata):
    FPS = 10
    ROBOT_TYPE = "franka"
    FEATURES = {
        "observation.images.egoview": {
            "dtype": "video",
            "shape": [1408, 1408, 3],
            "names": ["height", "width", "channel"],
            "video_info": {"video.fps": FPS, "video.codec": "h264"},
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (63,) 
        },
        "action": {
            "dtype": "float32",
            "shape": (63,)
        },
        "timestamp": {
            "dtype": "float32",
            "shape": [],
        },
    }
    return FPS, ROBOT_TYPE, FEATURES

def load_take_video_info(takes_path: Path) -> dict[str, dict]:
    with open(takes_path) as f:
        takes = json.load(f)

    take_map = {}
    for take in takes:
        uid = take["take_uid"]
        task_name = take["task_name"]
        found = False

        # 모든 카메라를 순회
        for cam_views in take.get("frame_aligned_videos", {}).values():
            for stream in cam_views.values():
                rel_path = stream.get("relative_path", "")
                if rel_path.endswith("214-1.mp4"):
                    take_map[uid] = {
                        "task_name": task_name,
                        "video_rel_path": Path(take["root_dir"]) / rel_path
                    }
                    found = True
                    break
            if found:
                break

        if not found:
            print(f"[WARN] UID {uid} has no video ending in 214-1.mp4")

    return take_map

def egoexo_to_lerobot(json_path: Path, video_path: Path, output_path: Path, task_name: str = "egopose") -> list[Path]:
    episode_dirs = []

    if not video_path.exists():
        print(f"[WARN] Video file not found: {video_path}")
        return episode_dirs

    if output_path.exists():
        print(f"[INFO] Removing existing directory: {output_path}")
        shutil.rmtree(output_path, ignore_errors=True)
        for _ in range(20):
            if not output_path.exists():
                print(f"[DEBUG] Successfully deleted {output_path}")
                break
            print(f"[DEBUG] Still exists... retrying delete")
            time.sleep(0.1)
            try:
                os.system(f"rm -rf '{output_path}'") 
            except Exception as e:
                print(f"[ERROR] rm -rf failed: {e}")
        else:
            raise RuntimeError(f"[ERROR] Failed to delete {output_path} after retries")

    with open(json_path) as f:
        anno = json.load(f)

    video = imageio.get_reader(str(video_path), format="mp4")
    FPS, robot_type, features = validate_all_metadata([])
    ds = LeRobotDataset.create(
        repo_id=json_path.stem,
        root=output_path,
        fps=FPS,
        robot_type=robot_type,
        features=features,
    )
    episode_dirs.append(output_path) 

    print("ds.features", ds.features)
    print("\n📦 Full ds.features structure:\n")
    for key, value in ds.features.items():
        print(f"{key}:")
        for sub_key, sub_val in value.items():
            print(f"  {sub_key}: {sub_val}")
        print()

    expected_diff = 3.0 / FPS
    prev_frame_id = None
    prev_state = None
    frame_added = False
    interval = 3
    interval_FPS = FPS * interval
    episode_count = 0

    for i, (frame_id, frame_data) in enumerate(sorted(anno.items(), key=lambda x: int(x[0]))):
        frame_id_int = int(frame_id)
        frame_data = frame_data[0]  

        try:
            image = video.get_data(frame_id_int)
        except IndexError:
            print(f"[WARN] Frame {frame_id_int} out of video bounds.")
            break

        state = extract_observation_state_from_3d(frame_data["annotation3D"], joints_order)
        timestamp = round(frame_id_int / interval_FPS, 6)

        if prev_frame_id is not None:
            gap = frame_id_int - prev_frame_id
            if gap == interval:
                pass  
            elif gap % interval == 0:
                print(f"[INFO] Interpolating {gap - interval} missing frames between {prev_frame_id} and {frame_id_int}")
                for interp_id in range(interval, gap, interval):
                    mid_frame_id = prev_frame_id + interval
                    mid_timestamp = round(mid_frame_id / interval_FPS, 6)
                    alpha = interp_id / gap
                    interp_state = (1 - alpha) * prev_state + alpha * state
                    dummy_image = video.get_data(mid_frame_id)

                    sample = {
                        "observation.images.egoview": dummy_image,
                        "observation.state": interp_state,
                        "action": interp_state,
                    }
                    ds.add_frame(sample, task=task_name, timestamp=mid_timestamp)
                    frame_added = True

                    prev_frame_id = mid_frame_id
                    prev_state = interp_state
                prev_frame_id += interval

            else:
              
                print(f"[NEW EPISODE] Irregular gap={gap} at frame {frame_id_int}")
                ds.save_episode()

                new_output_path = output_path.parent / f"{json_path.stem}_ep{episode_count:03d}"
                print(f"[INFO] Creating new episode directory: {new_output_path}")
                episode_count += 1
                ds = LeRobotDataset.create(
                    repo_id=f"{json_path.stem}_{episode_count:06}",
                    root=new_output_path,
                    fps=FPS,
                    robot_type=robot_type,
                    features=features,
                )
                episode_dirs.append(new_output_path)  
                frame_id = 0  


        sample = {
            "observation.images.egoview": image.astype(np.uint8),
            "observation.state": state,
            "action": state,
        }
        print("=============================================")
        print("[debug]timestamp:", timestamp)
        ds.add_frame(sample, task=task_name, timestamp=timestamp)
        frame_added = True

        prev_frame_id = frame_id_int
        prev_state = state

    if not frame_added:
        print(f"[WARN] No frames were added for {json_path.name}, skipping episode save.")
        return episode_dirs

    ds.save_episode()
    print(f"[✅] Episode saved for {json_path.name}")
    return episode_dirs


def aggregate_lerobot_datasets(temp_dirs: list[Path], aggregated_dir: Path):
    all_metadata = [LeRobotDatasetMetadata("", root=raw_dir) for raw_dir in temp_dirs]
    fps, robot_type, features = validate_all_metadata(all_metadata)

    if aggregated_dir.exists():
        shutil.rmtree(aggregated_dir)

    aggr_meta = LeRobotDatasetMetadata.create(
        repo_id=f"{aggregated_dir.parent.name}/{aggregated_dir.name}",
        root=aggregated_dir,
        fps=fps,
        robot_type=robot_type,
        features=features,
    )

    datasets_task_index_to_aggr_task_index = {}
    aggr_task_index = 0
    for dataset_index, meta in enumerate(all_metadata):
        task_index_to_aggr_task_index = {}
        for task_index, task in meta.tasks.items():
            if task not in aggr_meta.task_to_task_index:
                aggr_meta.tasks[aggr_task_index] = task
                aggr_meta.task_to_task_index[task] = aggr_task_index
                aggr_task_index += 1
            task_index_to_aggr_task_index[task_index] = aggr_meta.task_to_task_index[task]
        datasets_task_index_to_aggr_task_index[dataset_index] = task_index_to_aggr_task_index

    datasets_aggr_episode_index_shift = {}
    datasets_aggr_index_shift = {}

    for dataset_index, meta in enumerate(all_metadata):
        aggr_index_shift = aggr_meta.total_frames
        aggr_episode_index_shift = aggr_meta.total_episodes
        datasets_aggr_episode_index_shift[dataset_index] = aggr_episode_index_shift
        datasets_aggr_index_shift[dataset_index] = aggr_index_shift

        for episode_index, episode_dict in meta.episodes.items():
            aggr_episode_index = episode_index + aggr_episode_index_shift
            episode_dict["episode_index"] = aggr_episode_index
            aggr_meta.episodes[aggr_episode_index] = episode_dict

            df_path = meta.root / meta.get_data_file_path(episode_index)
            aggr_df_path = aggr_meta.root / aggr_meta.get_data_file_path(aggr_episode_index)
            aggr_df_path.parent.mkdir(parents=True, exist_ok=True)
            df = pd.read_parquet(df_path)
            df["index"] += aggr_index_shift
            df["episode_index"] += aggr_episode_index_shift
            df["task_index"] = df["task_index"].map(datasets_task_index_to_aggr_task_index[dataset_index])
            df.to_parquet(aggr_df_path)
        
        for episode_index in range(meta.total_episodes):
            aggr_episode_index = episode_index + aggr_episode_index_shift
            for vid_key in meta.video_keys:
                src_video = meta.root / meta.get_video_file_path(episode_index, vid_key)

                dst_video = aggr_meta.root / "videos" / f"episode_{aggr_episode_index:06d}.mp4"
                dst_video.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src_video, dst_video)

        for episode_index, episode_stats in meta.episodes_stats.items():
            aggr_episode_index = episode_index + aggr_episode_index_shift
            episode_stats["index"]["min"] += aggr_index_shift
            episode_stats["index"]["max"] += aggr_index_shift
            episode_stats["index"]["mean"] += aggr_index_shift
            episode_stats["episode_index"]["min"] += aggr_episode_index_shift
            episode_stats["episode_index"]["max"] += aggr_episode_index_shift
            episode_stats["episode_index"]["mean"] += aggr_episode_index_shift
            aggr_meta.episodes_stats[aggr_episode_index] = episode_stats

        aggr_meta.info["total_episodes"] += meta.total_episodes
        aggr_meta.info["total_frames"] += meta.total_frames
        aggr_meta.info["total_videos"] += len(aggr_meta.video_keys) * meta.total_episodes

    aggr_meta.info["total_tasks"] = len(aggr_meta.tasks)
    aggr_meta.info["total_chunks"] = aggr_meta.get_episode_chunk(aggr_meta.total_episodes - 1) + 1
    aggr_meta.info["splits"] = {"train": f"0:{aggr_meta.info['total_episodes']}"}


    for episode_dict in aggr_meta.episodes.values():
        write_episode(episode_dict, aggr_meta.root)
    for episode_index, episode_stats in aggr_meta.episodes_stats.items():
        write_episode_stats(episode_index, episode_stats, aggr_meta.root)
    for task_index, task in aggr_meta.tasks.items():
        write_task(task_index, task, aggr_meta.root)
    write_info(aggr_meta.info, aggr_meta.root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-dir", type=Path, required=True, help="Directory containing EgoExo JSONs")
    parser.add_argument("--video-dir", type=Path, required=True, help="Directory containing EgoExo MP4s")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory root")
    parser.add_argument("--aggregate-name", type=str, default="aggregated_lerobot", help="Name of aggregated output folder")
    parser.add_argument("--takes-path", type=Path, required=True, help="Path to takes.json file")
    args = parser.parse_args()
    takes_info = load_take_video_info(args.takes_path)
    temp_dirs = []  
    for json_path in sorted(args.json_dir.glob("*.json")):
        uid = json_path.stem
        take = takes_info.get(uid)
        print(f"[DEBUG] Processing JSON file: {json_path.name}, UID: {uid}")

        if take is None:
            print(f"[WARN] UID {uid} not found or invalid in takes.json")
            continue

        video_path = args.video_dir / take["video_rel_path"]
        if not video_path.exists():
            print(f"[WARN] Video file not found: {video_path}")
            continue

        
        temp_output = args.output_dir / f"{uid}_temp"
        related_output = sorted(args.output_dir.glob(f"{uid}_ep*"))


        if temp_output.exists():
            print(f"[SKIP] Temp output already exists for UID {uid}, skipping.")
            temp_dirs.append(temp_output)
            continue


        if related_output:
            print(f"[SKIP] Related output already exists for UID {uid}, skipping.")
            temp_dirs.extend(related_output)
            continue
        task_name = take["task_name"]  
        episode_dirs = egoexo_to_lerobot(json_path, video_path, temp_output, task_name)
        temp_dirs.extend(episode_dirs)
    aggregated_dir = args.output_dir / args.aggregate_name
    aggregate_lerobot_datasets(temp_dirs, aggregated_dir)

    for temp in temp_dirs:
        shutil.rmtree(temp, ignore_errors=True)
        print(f"[INFO] Deleted temporary folder: {temp}")