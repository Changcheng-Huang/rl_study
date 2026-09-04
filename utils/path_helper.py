import os

def get_manim_video_path(folder_name=None, quality="1080p60", video_file=None, scene_name=None):
    """
    更鲁棒的 Manim 视频路径获取：
    1) 优先使用 folder_name + quality + video_file 拼出标准路径
    2) 若找不到，则在 folder_name（或 scene_name）目录下递归寻找 mp4（排除 partial_movie_files）
    3) 支持 DQN 这种特殊目录：media/videos/DQN 1080p60/DQNDemo.mp4

    参数：
    - folder_name: media/videos 下的目录名（例如 Policy_iteration、QLearning、SARSA、Value_iteration、DQN 1080p60）
    - quality: 默认 1080p60
    - video_file: mp4 文件名（例如 PolicyIteration.mp4）
    - scene_name: 兼容旧接口：如果你只传 scene_name，会当成 folder_name 使用
    """
    if folder_name is None and scene_name is not None:
        folder_name = scene_name

    if folder_name is None:
        return None

    # 统一基路径
    base_dir = os.path.join("media", "videos", folder_name)

    # 情况 1：标准结构 media/videos/<folder>/<quality>/<file>.mp4
    if video_file is not None:
        candidate = os.path.join(base_dir, quality, video_file)
        if os.path.exists(candidate):
            return candidate

    # 情况 2：DQN 特殊结构 media/videos/<folder>/<file>.mp4（folder 可能就叫 "DQN 1080p60"）
    if video_file is not None:
        candidate2 = os.path.join(base_dir, video_file)
        if os.path.exists(candidate2):
            return candidate2

    # 情况 3：自动搜索：在 base_dir 下递归找 mp4，跳过 partial_movie_files，返回最大文件（通常是最终成片）
    if os.path.isdir(base_dir):
        mp4s = []
        for root, dirs, files in os.walk(base_dir):
            if "partial_movie_files" in root:
                continue
            for f in files:
                if f.lower().endswith(".mp4"):
                    mp4s.append(os.path.join(root, f))
        if mp4s:
            return max(mp4s, key=lambda p: os.path.getsize(p))

    print(f"Warning: Video not found under {base_dir}")
    return None
