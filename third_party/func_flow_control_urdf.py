import h5py
import numpy as np
import torch
import os
from typing import List, Dict, Optional, Tuple
from depth_anything_3.api import DepthAnything3
from scipy.ndimage import zoom, binary_dilation, distance_transform_edt
import cv2
import roboticstoolbox as rtb
from scipy.spatial import KDTree
import trimesh


def _silent_print(*args, **kwargs):
    return


class CondGenerator:
    def __init__(
            self,
            model_path: Optional[str] = None,
            urdf_path: Optional[str] = None,
            gripper_mesh_dir: Optional[str] = None,
            device: str = "cuda",
            arm_gray_threshold: int = 45,
            arm_v_threshold: int = 70,
            arm_s_threshold: int = 100,
            arm_dilate_iterations: int = 3,
            rendermask_dilate_iterations: int = 3,
            arm_sample_count: int = 3000,
            gpu_dist_chunk_size: int = 1024,
            load_da3_model: bool = False,
    ):
        """
        初始化条件生成器

        Args:
            model_path: DA3模型路径
            urdf_path: 机器人URDF文件路径
            gripper_mesh_dir: gripper STL文件目录
            device: 计算设备
            arm_gray_threshold: 灰度阈值，低于此值判定为机械臂像素（越大越激进）
            arm_v_threshold: HSV 亮度(V)阈值，低于此值判定为暗区（越大越激进）
            arm_s_threshold: HSV 饱和度(S)阈值，低于此值判定为无色（越大越激进）
            arm_dilate_iterations: mask 膨胀迭代次数（越大边缘过滤越多）
            rendermask_dilate_iterations: simulator rendermask 膨胀迭代次数，
                用 7x7 kernel，补偿 sim-to-real 对齐间隙（默认 3）
            arm_sample_count: 机械臂 body link (link1-link6) 每个 STL mesh 的采样点数
            gpu_dist_chunk_size: GPU 最近邻距离分块大小
            load_da3_model: 初始化时是否加载 DA3 权重；False 时在 forward_DA3 首次调用时延迟加载
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_path = model_path or os.environ.get("DA3_MODEL_PATH")
        self.model = None
        _silent_print(f"🚀 初始化条件生成器")
        _silent_print(f"  设备: {self.device}")

        if load_da3_model:
            self._load_da3_model()
        else:
            _silent_print("✓ DA3模型延迟加载")

        # 加载机器人运动学
        self.urdf_path = urdf_path or os.environ.get("PIPER_URDF_PATH")
        self.gripper_mesh_dir = gripper_mesh_dir or os.environ.get("PIPER_GRIPPER_MESH_DIR")
        if not self.urdf_path or not self.gripper_mesh_dir:
            raise ValueError(
                "Piper assets are required. Set PIPER_URDF_PATH and "
                "PIPER_GRIPPER_MESH_DIR, or pass urdf_path/gripper_mesh_dir."
            )
        self.arm_sample_count = arm_sample_count
        self.setup_robot_kinematics()

        # 设置相机参数
        self.setup_camera_params()

        # 碰撞检测阈值 (单位: 米)
        self.collision_threshold = 0.01
        self.scene_density_threshold = 20
        # 适用于 modality = 3D ，此时gripper无噪声; modality = 2D 时gripper有噪声
        self.gripper_contact_min_points = 1

        # 机械臂颜色过滤参数（仅影响颜色过滤分支，不影响 flow mask 分支）
        self.arm_gray_threshold = arm_gray_threshold
        self.arm_v_threshold = arm_v_threshold
        self.arm_s_threshold = arm_s_threshold
        self.arm_dilate_iterations = arm_dilate_iterations

        # Simulator rendermask 膨胀参数
        self.rendermask_dilate_iterations = rendermask_dilate_iterations
        self.gpu_dist_chunk_size = int(gpu_dist_chunk_size)

        # Scene flow 模型（延迟加载）
        self.scene_flow_model = None

        _silent_print(f"✓ 初始化完成\n")

    def _load_da3_model(self):
        if self.model is not None:
            return
        if not self.model_path:
            raise ValueError("DA3 model is required. Set DA3_MODEL_PATH or pass model_path.")
        self.model = DepthAnything3.from_pretrained(self.model_path)
        self.model = self.model.to(device=self.device)
        self.model.eval()
        _silent_print(f"✓ DA3模型加载完成")

    def setup_robot_kinematics(self):
        """
        设置机器人运动学链（纯 FK 版本）

        仅依赖:
          - T_front2base_left / T_front2base_right (front camera → 各臂 base 的标定矩阵)
          - piper_twin.urdf + STL meshes
        不依赖:
          - T_coord_* / T_gripper_* / T_cam2gripper_* 等手标修正矩阵
          - projection book / GripperGTMaskProvider
        """
        _silent_print(f"🤖 加载机器人URDF: {self.urdf_path}")

        # 加载机器人
        links, name, urdf_string, urdf_filepath = rtb.Robot.URDF_read(self.urdf_path)
        robot = rtb.Robot(links, name=name, manufacturer="Piper",
            urdf_string=urdf_string, urdf_filepath=urdf_filepath)

        # 创建运动学链（左右臂共享同一 URDF 拓扑，FK 时传不同 joint vector）
        self.ets_camera = robot.ets(end="camera")  # wrist camera pose
        self.ets_link7 = robot.ets(end="link7")  # gripper link7
        self.ets_link8 = robot.ets(end="link8")  # gripper link8

        # ETS chains + meshes for arm body links (link1-link6, for full robot FK)
        self.arm_body_link_names = []
        self.ets_arm_body = {}
        self.arm_body_mesh_pts = {}
        for link_name in ['link1', 'link2', 'link3', 'link4', 'link5', 'link6']:
            try:
                ets = robot.ets(end=link_name)
            except Exception:
                continue
            pts = self._load_link_mesh_optional(link_name, self.arm_sample_count)
            if pts is not None:
                self.arm_body_link_names.append(link_name)
                self.ets_arm_body[link_name] = ets
                self.arm_body_mesh_pts[link_name] = pts
        _silent_print(f"  机械臂 body links: {self.arm_body_link_names}")

        # SAPIEN camera → OpenCV camera 坐标转换
        # SAPIEN: X=forward, Y=left, Z=up → OpenCV: X=right, Y=down, Z=forward
        self.T_sapien2cv = np.array([
            [0., -1., 0., 0.],
            [0., 0., -1., 0.],
            [1., 0., 0., 0.],
            [0., 0., 0., 1.],
        ])

        # Front camera → 左/右臂 base 的标定矩阵（唯一需要的外部标定）
        self.T_front2base_left = np.array([
            [0.05831506, -0.84520743, 0.53124736, 0.02381213],
            [-0.99752094, -0.02833829, 0.06441209, -0.34711892],
            [-0.03938694, -0.53368656, -0.84476466, 0.66712113],
            [0, 0, 0, 1]
        ])

        self.T_front2base_right = np.array([
            [-0.00946253, -0.84779082, 0.53024634, 0.01105757],
            [-0.99886042, 0.03282072, 0.03465061, 0.25614093],
            [-0.04677953, -0.52931420, -0.84713527, 0.63708367],
            [0, 0, 0, 1]
        ])

        # 预计算逆矩阵（base → front）
        self.T_base2front_left = np.linalg.inv(self.T_front2base_left)
        self.T_base2front_right = np.linalg.inv(self.T_front2base_right)

        # Gripper qpos 值域映射常量: HDF5 真实值 → URDF prismatic joint 值
        self.GRIPPER_RAW_CLOSE = -0.001260
        self.GRIPPER_RAW_OPEN = 0.037064
        self.GRIPPER_JOINT_MIN = 0.0
        self.GRIPPER_JOINT_MAX = 0.04

        # 缓存 gripper STL mesh 点云
        self.mesh_link7_pts = self._load_gripper_mesh("link7")  # [5000, 3]
        self.mesh_link8_pts = self._load_gripper_mesh("link8")  # [5000, 3]

        _silent_print(f"✓ 运动学链加载完成（纯 FK，无修正矩阵）")

    def setup_camera_params(self):
        """设置相机参数"""
        # RGB相机内参
        self.intrinsics = {
            'left': np.array([
                [605.4948120117188, 0.0, 325.0260925292969],
                [0.0, 605.5114135742188, 246.6322479248047],
                [0.0, 0.0, 1.0]
            ]),
            'right': np.array([
                [607.4896850585938, 0.0, 332.4833984375],
                [0.0, 606.8885498046875, 249.5357666015625],
                [0.0, 0.0, 1.0]
            ]),
            'front': np.array([
                [488.615234375, 0.0, 321.0052185058594],
                [0.0, 488.615234375, 217.4329071044922],
                [0.0, 0.0, 1.0]
            ])
        }

        self.camera_names = ['front', 'left', 'right']

    def _convert_gripper_qpos(self, qpos_14: np.ndarray) -> np.ndarray:
        """
        将 HDF5 真实 gripper qpos 转换为 URDF prismatic joint 值。

        转换: raw → [0,1] 归一化 → [JOINT_MIN, JOINT_MAX] 映射
        """
        q = qpos_14.copy().astype(np.float64)
        for idx in [6, 13]:
            raw = q[idx]
            norm = np.clip(
                (raw - self.GRIPPER_RAW_CLOSE) / (self.GRIPPER_RAW_OPEN - self.GRIPPER_RAW_CLOSE),
                0, 1
            )
            q[idx] = self.GRIPPER_JOINT_MIN + norm * (self.GRIPPER_JOINT_MAX - self.GRIPPER_JOINT_MIN)
        return q

    def _arm_q7_to_q8(self, q_arm_7: np.ndarray) -> np.ndarray:
        """
        将 7 元素 arm qpos 扩展为 8 元素 (j8 mimic j7)。

        URDF 有 8 个 active joints: j1-j6(revolute) + j7,j8(prismatic)。
        j8 mimic j7。HDF5 每臂只有 7 个值，需要复制 gripper 给 j7 和 j8，
        然后各除以 2（两指各开一半）。
        """
        q8 = np.zeros(8, dtype=np.float64)
        q8[:7] = q_arm_7
        q8[7] = q_arm_7[6]  # j8 = j7
        q8[6:] /= 2
        return q8

    def _load_gripper_mesh(self, link_name: str) -> np.ndarray:
        """
        加载gripper的STL网格并转换为点云

        Args:
            link_name: "link7" 或 "link8"

        Returns:
            points: [N, 3] numpy数组，网格顶点坐标
        """
        mesh_path = os.path.join(self.gripper_mesh_dir, f"{link_name}.STL")
        if not os.path.exists(mesh_path):
            raise FileNotFoundError(f"Gripper mesh文件不存在: {mesh_path}")

        # 加载STL网格
        mesh = trimesh.load(mesh_path)

        # 在表面均匀采样更多点（推荐，点云更密集）
        points, _ = trimesh.sample.sample_surface(mesh, count=5000)

        return points

    def _load_link_mesh_optional(self, link_name: str, sample_count: int) -> Optional[np.ndarray]:
        """加载 link 的 STL mesh 并采样点云，不存在则返回 None"""
        for ext in ['.STL', '.stl']:
            path = os.path.join(self.gripper_mesh_dir, f"{link_name}{ext}")
            if os.path.exists(path):
                mesh = trimesh.load(path)
                pts, _ = trimesh.sample.sample_surface(mesh, count=sample_count)
                return pts.astype(np.float64)
        return None

    def forward_kinematics(self, current_action: np.ndarray) -> Dict[str, np.ndarray]:
        """
        通过纯 FK 计算三视角 W2C 外参（无修正矩阵）。

        流程: URDF camera link FK → SAPIEN→OpenCV 转换 → W2C

        Args:
            current_action: 当前时刻的qpos [14,] (左臂7 + 右臂7)

        Returns:
            extrinsics: 字典，包含三个相机的W2C外参
        """
        qpos = self._convert_gripper_qpos(current_action)
        q_left = self._arm_q7_to_q8(qpos[:7])
        q_right = self._arm_q7_to_q8(qpos[7:])

        extrinsics = {'front': np.eye(4)}

        # 左臂 wrist camera
        T_base_cam_L = self.ets_camera.fkine(q_left).A  # cam_local → base
        T_front_cam_L = self.T_base2front_left @ T_base_cam_L  # cam → world (SAPIEN 约定)
        extrinsics['left'] = self.T_sapien2cv @ np.linalg.inv(T_front_cam_L)  # W2C (OpenCV)

        # 右臂 wrist camera
        T_base_cam_R = self.ets_camera.fkine(q_right).A
        T_front_cam_R = self.T_base2front_right @ T_base_cam_R
        extrinsics['right'] = self.T_sapien2cv @ np.linalg.inv(T_front_cam_R)

        return extrinsics

    def forward_DA3(
            self,
            current_obs: List[np.ndarray],
            extrinsics: Dict[str, np.ndarray]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        运行DA3推理获取深度图

        Args:
            current_obs: 三视角RGB图像列表 [front_rgb, left_rgb, right_rgb]
            extrinsics: 三视角外参字典

        Returns:
            depths: 深度图字典
            extrinsics: 外参字典
            intrinsics: 内参字典
        """
        self._load_da3_model()

        # 准备输入
        images_array = []
        intrinsics_list = []
        extrinsics_list = []

        for cam in self.camera_names:
            idx = self.camera_names.index(cam)
            images_array.append(current_obs[idx])
            intrinsics_list.append(self.intrinsics[cam])
            extrinsics_list.append(extrinsics[cam])

        intrinsics_array = np.stack(intrinsics_list, axis=0)
        extrinsics_array = np.stack(extrinsics_list, axis=0)

        # DA3推理
        with torch.inference_mode():
            prediction = self.model.inference(
                image=images_array,
                intrinsics=intrinsics_array,
                extrinsics=extrinsics_array,
                use_ray_pose=True,
                infer_gs=False
            )

        # 组织结果
        depths = {}
        extrinsics_out = {}
        intrinsics_out = {}

        for i, cam in enumerate(self.camera_names):
            depths[cam] = prediction.depth[i]
            extrinsics_out[cam] = prediction.extrinsics[i]
            intrinsics_out[cam] = prediction.intrinsics[i]

        return depths, extrinsics_out, intrinsics_out

    def convert_depth(
            self,
            current_obs: List[np.ndarray],
            depths: Dict[str, np.ndarray],
            extrinsics: Dict[str, np.ndarray],
            intrinsics: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        将深度图转换为点云（世界坐标系 - front camera）

        Returns:
            front_pts, left_pts, right_pts: 三个相机的点云
            arm_masks: Dict[str, np.ndarray]，每个相机的机械臂mask
        """
        points_dict = {}
        arm_masks = {}

        for i, cam in enumerate(self.camera_names):
            rgb = current_obs[i]
            depth = depths[cam]
            intrinsic = intrinsics[cam]
            extrinsic = extrinsics[cam]

            if extrinsic.shape == (3, 4):
                extrinsic = np.vstack([extrinsic, [0, 0, 0, 1]])

            mask_arm = self._detect_arm_pixels(rgb)
            arm_masks[cam] = mask_arm

            _silent_print(f"  {cam}: 机械臂像素 {mask_arm.sum()} / {mask_arm.size} "
                  f"({mask_arm.sum() / mask_arm.size * 100:.1f}%)")

            points = self._depth_to_pointcloud(
                depth=depth,
                intrinsic=intrinsic,
                rgb=rgb,
                mask_exclude=mask_arm
            )

            if cam != 'front':
                points = self._transform_to_world(points, extrinsic)

            points_dict[cam] = points

        return points_dict['front'], points_dict['left'], points_dict['right'], arm_masks

    def convert_depth_with_flow_mask(
            self,
            current_obs: List[np.ndarray],
            depths: Dict[str, np.ndarray],
            extrinsics: Dict[str, np.ndarray],
            intrinsics: Dict[str, np.ndarray],
            flow_lg_2D: Dict[str, np.ndarray],
            flow_rg_2D: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray],
               List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        深度图转点云 + 基于 flow mask 重建 K+1 帧 gripper 点云（2D 对齐专用）

        与 convert_depth 的区别:
        - front camera: 仍用颜色过滤去除机械臂像素（gripper 被机械臂遮挡不可见）
        - left/right wrist camera: 使用上游 flow mask 区分场景 vs gripper 像素
          - 场景像素 → 转为场景点云
          - gripper 像素 → 利用 wrist depth 反投影为 T=0 gripper 点云，
            再结合 3D flow 得到 K+1 帧 gripper 点云

        输出格式与 3D 方案的 get_gripper_points 完全一致:
        每个 gripper 返回 List[np.ndarray]，长度 K+1，每个 [N, 3]，
        均在 front camera 坐标系（世界坐标系）下。
        注意: 2D 方案 link7+link8 合并投影后对半拆分，前半 → lg1/rg1，后半 → lg2/rg2。

        Args:
            current_obs: 三视角 RGB [front, left, right]
            depths: 深度图字典 {'front', 'left', 'right'}
            extrinsics: 外参字典
            intrinsics: 内参字典
            flow_lg_2D: 左臂 gripper (link7+link8 合并) 的 2D flow,
                        dict with 'mask' [H,W] bool, 'flow' [H,W,K,3]
            flow_rg_2D: 右臂 gripper (link7+link8 合并) 的 2D flow

        Returns:
            front_pts: front camera 场景点云 [N, 6] (xyz+rgb)
            left_pts: left camera 场景点云（去除 gripper 区域）[N, 6]
            right_pts: right camera 场景点云（去除 gripper 区域）[N, 6]
            arm_masks: 机械臂 mask 字典
            gripper_pts_lg1: List[np.ndarray], 左臂 link7 点云, 长度 K+1, 每个 [N1, 3]
            gripper_pts_lg2: List[np.ndarray], 左臂 link8 点云, 长度 K+1, 每个 [N2, 3]
            gripper_pts_rg1: List[np.ndarray], 右臂 link7 点云, 长度 K+1, 每个 [N1, 3]
            gripper_pts_rg2: List[np.ndarray], 右臂 link8 点云, 长度 K+1, 每个 [N2, 3]

        """
        K = flow_lg_2D['flow'].shape[2]

        points_dict = {}
        arm_masks = {}

        # --- Front camera: 颜色过滤机械臂（与 convert_depth 相同）---
        rgb_front = current_obs[0]
        depth_front = depths['front']
        intr_front = intrinsics['front']
        ext_front = extrinsics['front']
        if ext_front.shape == (3, 4):
            ext_front = np.vstack([ext_front, [0, 0, 0, 1]])

        mask_arm_front = self._detect_arm_pixels(rgb_front)
        arm_masks['front'] = mask_arm_front

        front_pts = self._depth_to_pointcloud(
            depth_front, intr_front, rgb_front, mask_exclude=mask_arm_front
        )
        # front = world，无需变换
        points_dict['front'] = front_pts

        _silent_print(f"  front: 场景点 {len(front_pts)}, "
              f"机械臂像素 {mask_arm_front.sum()} / {mask_arm_front.size} "
              f"({mask_arm_front.sum() / mask_arm_front.size * 100:.1f}%)")

        # --- Left wrist camera: flow mask 区分场景/gripper ---
        rgb_left = current_obs[1]
        depth_left = depths['left']
        intr_left = intrinsics['left']
        ext_left = extrinsics['left']
        if ext_left.shape == (3, 4):
            ext_left = np.vstack([ext_left, [0, 0, 0, 1]])

        gripper_mask_left = flow_lg_2D['mask']
        arm_masks['left'] = gripper_mask_left

        # 场景点云：排除 gripper 区域（resize mask 到 depth 分辨率）
        H_depth_l, W_depth_l = depth_left.shape
        if gripper_mask_left.shape[:2] != (H_depth_l, W_depth_l):
            gripper_mask_left_rs = cv2.resize(
                gripper_mask_left.astype(np.uint8), (W_depth_l, H_depth_l),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        else:
            gripper_mask_left_rs = gripper_mask_left

        left_scene_pts = self._depth_to_pointcloud(
            depth_left, intr_left, rgb_left, mask_exclude=gripper_mask_left_rs
        )
        left_scene_pts = self._transform_to_world(left_scene_pts, ext_left)
        points_dict['left'] = left_scene_pts

        # Gripper 点云：depth 反投影 + flow 重建 K+1 帧（按 label_map 拆分 link7/link8）
        C2W_left = np.linalg.inv(ext_left)
        gripper_pts_lg1, gripper_pts_lg2 = self._reconstruct_gripper_from_flow(
            depth_left, intr_left, C2W_left, flow_lg_2D, K
        )

        _silent_print(f"  left: 场景点 {len(left_scene_pts)}, "
              f"gripper像素 {gripper_mask_left.sum()}, "
              f"lg1(link7)点数 {len(gripper_pts_lg1[0])}, lg2(link8)点数 {len(gripper_pts_lg2[0])}")

        # --- Right wrist camera: 同 left ---
        rgb_right = current_obs[2]
        depth_right = depths['right']
        intr_right = intrinsics['right']
        ext_right = extrinsics['right']
        if ext_right.shape == (3, 4):
            ext_right = np.vstack([ext_right, [0, 0, 0, 1]])

        gripper_mask_right = flow_rg_2D['mask']
        arm_masks['right'] = gripper_mask_right

        H_depth_r, W_depth_r = depth_right.shape
        if gripper_mask_right.shape[:2] != (H_depth_r, W_depth_r):
            gripper_mask_right_rs = cv2.resize(
                gripper_mask_right.astype(np.uint8), (W_depth_r, H_depth_r),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        else:
            gripper_mask_right_rs = gripper_mask_right

        right_scene_pts = self._depth_to_pointcloud(
            depth_right, intr_right, rgb_right, mask_exclude=gripper_mask_right_rs
        )
        right_scene_pts = self._transform_to_world(right_scene_pts, ext_right)
        points_dict['right'] = right_scene_pts

        C2W_right = np.linalg.inv(ext_right)
        gripper_pts_rg1, gripper_pts_rg2 = self._reconstruct_gripper_from_flow(
            depth_right, intr_right, C2W_right, flow_rg_2D, K
        )

        _silent_print(f"  right: 场景点 {len(right_scene_pts)}, "
              f"gripper像素 {gripper_mask_right.sum()}, "
              f"rg1(link7)点数 {len(gripper_pts_rg1[0])}, rg2(link8)点数 {len(gripper_pts_rg2[0])}")

        return (points_dict['front'], points_dict['left'], points_dict['right'],
                arm_masks,
                gripper_pts_lg1, gripper_pts_lg2, gripper_pts_rg1, gripper_pts_rg2)

    def _detect_arm_pixels(self, rgb: np.ndarray) -> np.ndarray:
        """
        检测机械臂像素（HSV空间：低亮度 + 低饱和度的黑色区域）

        使用实例属性控制过滤强度:
            arm_gray_threshold: 灰度阈值（默认 45，越大越激进）
            arm_v_threshold:    HSV V 阈值（默认 70，越大越激进）
            arm_s_threshold:    HSV S 阈值（默认 100，越大越激进）
            arm_dilate_iterations: 膨胀次数（默认 3，越大边缘去得越干净）
        """
        # 转换为灰度图
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        # 转换为HSV图
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

        # 黑色机械臂: 低V（亮度）且低S（饱和度）
        # 这排除了深色但有颜色的物体（如深蓝桌面）
        mask_dark = hsv[:, :, 2] < self.arm_v_threshold
        mask_low_sat = hsv[:, :, 1] < self.arm_s_threshold

        mask_arm = (mask_dark & mask_low_sat) | (gray < self.arm_gray_threshold)

        # 形态学操作：去噪 + 填充 + 膨胀
        kernel = np.ones((5, 5), np.uint8)
        mask_arm = cv2.morphologyEx(mask_arm.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=2)
        mask_arm = cv2.morphologyEx(mask_arm, cv2.MORPH_OPEN, kernel, iterations=1)
        kernel = np.ones((3, 3), np.uint8)
        mask_arm = cv2.dilate(mask_arm, kernel, iterations=self.arm_dilate_iterations)

        return mask_arm.astype(bool)

    def _enhance_robot_mask(
            self,
            rendermask: np.ndarray,
            rgb: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        增强 simulator rendermask: 膨胀补偿 sim-to-real 间隙 + 可选颜色检测联合。

        策略:
          1. 先在 rendermask 原始分辨率做 7x7 kernel 膨胀
          2. 若提供 RGB，resize 到 RGB 分辨率后与颜色检测结果取并集

        Args:
            rendermask: [H, W] bool/uint8, simulator 渲染的 robot mask
            rgb: [H_rgb, W_rgb, 3] 可选 RGB（用于颜色辅助检测，补捕黑色部分）

        Returns:
            增强后的 bool mask（分辨率与 rendermask 一致，除非提供 rgb 则为 RGB 分辨率）
        """
        mask = rendermask.astype(np.uint8)

        # 膨胀: 7x7 kernel
        if self.rendermask_dilate_iterations > 0:
            kernel = np.ones((7, 7), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=self.rendermask_dilate_iterations)

        if rgb is not None:
            H_rgb, W_rgb = rgb.shape[:2]
            if mask.shape[:2] != (H_rgb, W_rgb):
                mask = cv2.resize(mask, (W_rgb, H_rgb), interpolation=cv2.INTER_NEAREST)
            color_mask = self._detect_arm_pixels(rgb).astype(np.uint8)
            mask = np.maximum(mask, color_mask)

        return mask.astype(bool)

    def _depth_to_pointcloud(
            self,
            depth: np.ndarray,
            intrinsic: np.ndarray,
            rgb: np.ndarray,
            mask_exclude: Optional[np.ndarray] = None,
            depth_range: Tuple[float, float] = (0.0, 5.0)
    ) -> np.ndarray:
        H, W = depth.shape

        # 调整RGB尺寸
        if rgb.shape[:2] != (H, W):
            scale_h = H / rgb.shape[0]
            scale_w = W / rgb.shape[1]
            rgb = zoom(rgb, (scale_h, scale_w, 1), order=1)

        # 调整mask尺寸
        if mask_exclude is not None and mask_exclude.shape[:2] != (H, W):
            mask_exclude = cv2.resize(
                mask_exclude.astype(np.uint8), (W, H),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)

        # 生成像素坐标
        u, v = np.meshgrid(np.arange(W), np.arange(H))

        # 有效性mask
        valid_mask = (depth > depth_range[0]) & (depth < depth_range[1])
        if mask_exclude is not None:
            valid_mask = valid_mask & (~mask_exclude)

        # 提取有效像素
        u_valid = u[valid_mask]
        v_valid = v[valid_mask]
        z_valid = depth[valid_mask]

        # 反投影
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]

        x = (u_valid - cx) * z_valid / fx
        y = (v_valid - cy) * z_valid / fy
        z = z_valid

        points_xyz = np.stack([x, y, z], axis=-1)
        colors = rgb[valid_mask].astype(np.float32) / 255.0

        points = np.concatenate([points_xyz, colors], axis=-1)

        return points

    def _transform_to_world(
            self,
            points: np.ndarray,
            extrinsic: np.ndarray
    ) -> np.ndarray:
        xyz = points[:, :3]
        colors = points[:, 3:]

        # DA3可能输出 (3,4)，补全为 (4,4)
        if extrinsic.shape == (3, 4):
            extrinsic = np.vstack([extrinsic, [0, 0, 0, 1]])

        C2W = np.linalg.inv(extrinsic)

        xyz_homo = np.concatenate([xyz, np.ones((len(xyz), 1))], axis=-1)
        xyz_world = (C2W @ xyz_homo.T).T[:, :3]

        return np.concatenate([xyz_world, colors], axis=-1)

    def _reconstruct_gripper_from_flow(
            self,
            depth: np.ndarray,
            intrinsic: np.ndarray,
            C2W: np.ndarray,
            flow_2D: Dict[str, np.ndarray],
            K: int,
            depth_range: Tuple[float, float] = (0.0, 5.0)
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        从 wrist depth + flow mask 重建 K+1 帧 gripper 点云（世界坐标系）

        流程:
            1. mask 内像素用 depth 反投影为 T=0 3D 点（camera 坐标系）
            2. C2W 变换到世界坐标系
            3. 逐帧加 3D flow 得到 T+k 点云
            4. 按 label_map (0=link7, 1=link8) 拆分为 g1/g2 两个列表

        Args:
            depth: [H_d, W_d] wrist camera 深度图（DA3 输出分辨率）
            intrinsic: [3, 3] DA3 输出的内参（与 depth 分辨率匹配）
            C2W: [4, 4] camera-to-world 变换矩阵
            flow_2D: dict with:
                'mask'      [H, W] bool
                'flow'      [H, W, K, 3]
                'label_map' [H, W] int8, -1/0/1  (由 get_flow_project_refine 生成)
            K: 未来帧数
            depth_range: 有效深度范围

        Returns:
            g1_pts_list: List[np.ndarray], 长度 K+1, 每个 [N1, 3], link7 点云, 世界坐标系
            g2_pts_list: List[np.ndarray], 长度 K+1, 每个 [N2, 3], link8 点云, 世界坐标系
        """
        empty = lambda: [np.zeros((0, 3), dtype=np.float64) for _ in range(K + 1)]

        H_depth, W_depth = depth.shape
        mask = flow_2D['mask']  # [H_img, W_img]
        flow = flow_2D['flow']  # [H_img, W_img, K, 3]
        label = flow_2D.get('label_map', None)  # [H_img, W_img] int8, 可能不存在
        H_img, W_img = mask.shape

        # 将 mask / flow / label 统一 resize 到 depth 分辨率
        if (H_depth, W_depth) != (H_img, W_img):
            mask_rs = cv2.resize(
                mask.astype(np.uint8), (W_depth, H_depth),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            flow_rs = np.zeros((H_depth, W_depth, K, 3), dtype=flow.dtype)
            for k in range(K):
                for c in range(3):
                    flow_rs[:, :, k, c] = cv2.resize(
                        flow[:, :, k, c], (W_depth, H_depth),
                        interpolation=cv2.INTER_LINEAR
                    )
            if label is not None:
                label_rs = cv2.resize(
                    label.astype(np.int8), (W_depth, H_depth),
                    interpolation=cv2.INTER_NEAREST
                ).astype(np.int8)
            else:
                label_rs = None
        else:
            mask_rs = mask
            flow_rs = flow
            label_rs = label

        # 获取 mask 内像素坐标
        v_idx, u_idx = np.where(mask_rs)
        if len(v_idx) == 0:
            return empty(), empty()

        # 深度有效性过滤
        z = depth[v_idx, u_idx].astype(np.float64)
        valid = (z > depth_range[0]) & (z < depth_range[1])
        v_idx, u_idx, z = v_idx[valid], u_idx[valid], z[valid]
        if len(z) == 0:
            return empty(), empty()

        # 反投影为 3D 点（camera 坐标系）→ 世界坐标系
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]
        x = (u_idx.astype(np.float64) - cx) * z / fx
        y = (v_idx.astype(np.float64) - cy) * z / fy
        pts_cam = np.stack([x, y, z], axis=-1)  # [N, 3]

        pts_homo = np.concatenate([pts_cam, np.ones((len(pts_cam), 1))], axis=-1)
        pts_world = (C2W @ pts_homo.T).T[:, :3]  # [N, 3]

        # 按 label_map 确定每个反投影点属于 link7(0) 还是 link8(1)
        if label_rs is not None:
            point_labels = label_rs[v_idx, u_idx]  # [N] int8
        else:
            # 无 label 信息时全归 g1，g2 为空（兜底）
            point_labels = np.zeros(len(v_idx), dtype=np.int8)

        g1_sel = (point_labels == 0)
        g2_sel = (point_labels == 1)

        # T=0 基础点云（按 link 拆分）
        g1_t0 = pts_world[g1_sel]
        g2_t0 = pts_world[g2_sel]
        g1_list = [g1_t0.copy()]
        g2_list = [g2_t0.copy()]

        # 逐帧加 3D flow
        for k in range(K):
            flow_k = flow_rs[v_idx, u_idx, k, :].astype(np.float64)  # [N, 3]
            pts_k = pts_world + flow_k
            g1_list.append(pts_k[g1_sel])
            g2_list.append(pts_k[g2_sel])

        return g1_list, g2_list

    def get_gripper_points(
            self,
            current_future_action: np.ndarray
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        获取当前帧+未来K帧的gripper点云（共K+1帧），纯 FK 版本。

        Args:
            current_future_action: 动作序列 [K+1, 14]，包含T到T+K时刻

        Returns:
            gripper_pts_lg1_list: 左臂link7点云列表，长度K+1，每个 [5000, 3]，世界坐标系
            gripper_pts_lg2_list: 左臂link8点云列表
            gripper_pts_rg1_list: 右臂link7点云列表
            gripper_pts_rg2_list: 右臂link8点云列表
        """
        num_frames = len(current_future_action)

        gripper_pts_lg1_list = []
        gripper_pts_lg2_list = []
        gripper_pts_rg1_list = []
        gripper_pts_rg2_list = []

        for t in range(num_frames):
            qpos = self._convert_gripper_qpos(current_future_action[t])
            q_left = self._arm_q7_to_q8(qpos[:7])
            q_right = self._arm_q7_to_q8(qpos[7:])

            # 左臂 link7/link8
            T_base_l7_L = self.ets_link7.fkine(q_left).A
            T_base_l8_L = self.ets_link8.fkine(q_left).A
            lg1 = self._transform_gripper_to_world(
                self.mesh_link7_pts, T_base_l7_L, self.T_front2base_left)
            lg2 = self._transform_gripper_to_world(
                self.mesh_link8_pts, T_base_l8_L, self.T_front2base_left)

            # 右臂 link7/link8
            T_base_l7_R = self.ets_link7.fkine(q_right).A
            T_base_l8_R = self.ets_link8.fkine(q_right).A
            rg1 = self._transform_gripper_to_world(
                self.mesh_link7_pts, T_base_l7_R, self.T_front2base_right)
            rg2 = self._transform_gripper_to_world(
                self.mesh_link8_pts, T_base_l8_R, self.T_front2base_right)

            gripper_pts_lg1_list.append(lg1)
            gripper_pts_lg2_list.append(lg2)
            gripper_pts_rg1_list.append(rg1)
            gripper_pts_rg2_list.append(rg2)

        return gripper_pts_lg1_list, gripper_pts_lg2_list, gripper_pts_rg1_list, gripper_pts_rg2_list

    def get_gripper_flow(
            self,
            gripper_pts_lg1: List[np.ndarray],
            gripper_pts_lg2: List[np.ndarray],
            gripper_pts_rg1: List[np.ndarray],
            gripper_pts_rg2: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        计算 gripper 3D flow: 当前帧 (T=0) 点云到未来每帧的位移

        对每个 gripper link:
            flow[i, k, :] = gripper_pts[k+1][i] - gripper_pts[0][i]
        即第 i 个点从 T=0 到 T=k+1 的 3D 位移向量。

        前提: 所有帧的点云来自同一 STL 采样，点索引一一对应。

        Args:
            gripper_pts_lg1: 左臂 gripper1 点云列表, 长度 K+1, 每个 [5000, 3]
            gripper_pts_lg2: 左臂 gripper2 点云列表
            gripper_pts_rg1: 右臂 gripper1 点云列表
            gripper_pts_rg2: 右臂 gripper2 点云列表

        Returns:
            flow_lg1: [5000, K, 3] 左臂 gripper1 的 3D flow
            flow_lg2: [5000, K, 3] 左臂 gripper2 的 3D flow
            flow_rg1: [5000, K, 3] 右臂 gripper1 的 3D flow
            flow_rg2: [5000, K, 3] 右臂 gripper2 的 3D flow

        Note:
            点云索引一一对应（同一 STL 采样），因此直接做差即可。
        """
        K = len(gripper_pts_lg1) - 1  # K+1 帧中有 K 个未来帧

        def _compute_flow(pts_list):
            pts_t0 = pts_list[0]  # [5000, 3]
            return np.stack([pts_list[k + 1] - pts_t0 for k in range(K)], axis=1)  # [5000, K, 3]

        flow_lg1 = _compute_flow(gripper_pts_lg1)
        flow_lg2 = _compute_flow(gripper_pts_lg2)
        flow_rg1 = _compute_flow(gripper_pts_rg1)
        flow_rg2 = _compute_flow(gripper_pts_rg2)

        _silent_print(f"  gripper 3D flow: shape={flow_lg1.shape}, K={K}")
        return flow_lg1, flow_lg2, flow_rg1, flow_rg2

    def get_flow_project_refine(
            self,
            flow_g1: np.ndarray,
            flow_g2: np.ndarray,
            pts_g1_world: np.ndarray,
            pts_g2_world: np.ndarray,
            extrinsic: np.ndarray,
            cam_name: str,
            image_size: Tuple[int, int] = (480, 640)
    ) -> Dict[str, np.ndarray]:
        """
        将 gripper 3D flow 投影到 camera 2D 像素空间（纯 FK W2C 版本）。

        流程:
            1. 拼接 link7+link8 的 T=0 世界坐标点和 flow 值
            2. 用 FK W2C 外参 + 内参投影到像素
            3. 写入 flow 值 + label (0=link7, 1=link8)
            4. 膨胀填充

        Args:
            flow_g1: [5000, K, 3] gripper link7 的 3D flow
            flow_g2: [5000, K, 3] gripper link8 的 3D flow
            pts_g1_world: [5000, 3] T=0 link7 世界坐标点
            pts_g2_world: [5000, 3] T=0 link8 世界坐标点
            extrinsic: [4, 4] W2C 外参（FK 计算）
            cam_name: 相机名称
            image_size: 图像尺寸 (H, W)

        Returns:
            flow_g_2D: dict with:
                'mask': [H, W] bool, gripper 像素 mask
                'flow': [H, W, K, 3] float, 每个像素的 3D flow
                'label_map': [H, W] int8, 0=link7, 1=link8, -1=非gripper
        """
        H, W = image_size
        K = flow_g1.shape[1]
        intrinsic = self.intrinsics[cam_name]

        # 拼接 link7 + link8
        pts_world = np.concatenate([pts_g1_world, pts_g2_world], axis=0)  # [N, 3]
        flow_combined = np.concatenate([flow_g1, flow_g2], axis=0)  # [N, K, 3]
        n7 = len(pts_g1_world)
        labels = np.concatenate([
            np.zeros(n7, dtype=np.int8),
            np.ones(len(pts_g2_world), dtype=np.int8),
        ])

        # W2C 投影
        if extrinsic.shape == (3, 4):
            extrinsic = np.vstack([extrinsic, [0, 0, 0, 1]])

        xyz_homo = np.concatenate([pts_world, np.ones((len(pts_world), 1))], axis=-1)
        xyz_cam = (extrinsic @ xyz_homo.T).T[:, :3]

        valid = xyz_cam[:, 2] > 0.01
        if valid.sum() == 0:
            flow_map = np.zeros((H, W, K, 3), dtype=np.float32)
            label_map = np.full((H, W), -1, dtype=np.int8)
            return {'mask': np.zeros((H, W), dtype=bool), 'flow': flow_map, 'label_map': label_map}

        X, Y, Z = xyz_cam[valid, 0], xyz_cam[valid, 1], xyz_cam[valid, 2]
        u = np.round(intrinsic[0, 0] * X / Z + intrinsic[0, 2]).astype(int)
        v = np.round(intrinsic[1, 1] * Y / Z + intrinsic[1, 2]).astype(int)
        flow_valid = flow_combined[valid].astype(np.float32)
        labels_valid = labels[valid]

        in_image = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u, v = u[in_image], v[in_image]
        flow_valid = flow_valid[in_image]
        labels_valid = labels_valid[in_image]

        flow_map = np.zeros((H, W, K, 3), dtype=np.float32)
        label_map = np.full((H, W), -1, dtype=np.int8)

        if len(u) == 0:
            return {'mask': np.zeros((H, W), dtype=bool), 'flow': flow_map, 'label_map': label_map}

        flow_map[v, u] = flow_valid
        label_map[v, u] = labels_valid

        # 膨胀 + 最近邻填充
        occ = np.zeros((H, W), dtype=bool)
        occ[v, u] = True
        struct = np.ones((3, 3), dtype=bool)
        dilated = binary_dilation(occ, structure=struct, iterations=2)

        _, nearest_idx = distance_transform_edt(~occ, return_indices=True)
        new_pixels = dilated & ~occ
        flow_map[new_pixels] = flow_map[nearest_idx[0][new_pixels], nearest_idx[1][new_pixels]]
        label_map[new_pixels] = label_map[nearest_idx[0][new_pixels], nearest_idx[1][new_pixels]]

        _silent_print(f"  {cam_name} link7+link8: 投影点 {valid.sum()}, mask像素 {dilated.sum()}, "
              f"link7像素 {(label_map == 0).sum()}, link8像素 {(label_map == 1).sum()}")

        return {'mask': dilated, 'flow': flow_map, 'label_map': label_map}

    def _transform_gripper_to_world(
            self,
            gripper_points: np.ndarray,
            T_gripper_to_base: np.ndarray,
            T_front_to_base: np.ndarray
    ) -> np.ndarray:
        """
        将gripper点云从gripper坐标系转换到世界坐标系

        Args:
            gripper_points: [N, 3] gripper局部坐标系中的点
            T_gripper_to_base: [4, 4] base -> gripper的变换
            T_front_to_base: [4, 4] front -> base的变换

        Returns:
            points_world: [N, 3] 世界坐标系中的点
        """
        # gripper -> base -> front
        T_base_to_front = np.linalg.inv(T_front_to_base)
        T_gripper_to_front = T_base_to_front @ T_gripper_to_base

        # 齐次坐标变换
        gripper_points_homo = np.concatenate([gripper_points,
                                              np.ones((len(gripper_points), 1))], axis=-1)
        points_world = (T_gripper_to_front @ gripper_points_homo.T).T[:, :3]

        return points_world

    def _compute_full_robot_points(self, qpos_14: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算双臂所有 link (link1-link8) 的点云在 front 坐标系下的位置。

        Returns:
            all_pts: [N_total, 3] 所有 link 点云（front 坐标系）
            gripper_mask: [N_total] bool, True 表示该点属于 gripper (link7/link8)
        """
        qpos = self._convert_gripper_qpos(qpos_14)
        q_left = self._arm_q7_to_q8(qpos[:7])
        q_right = self._arm_q7_to_q8(qpos[7:])

        all_pts = []
        all_is_gripper = []

        # arm body links (link1-link6)
        for link_name in self.arm_body_link_names:
            ets = self.ets_arm_body[link_name]
            mesh_pts = self.arm_body_mesh_pts[link_name]

            T_base_link_L = ets.fkine(q_left).A
            pts_L = self._transform_gripper_to_world(mesh_pts, T_base_link_L, self.T_front2base_left)
            all_pts.append(pts_L)
            all_is_gripper.append(np.zeros(len(pts_L), dtype=bool))

            T_base_link_R = ets.fkine(q_right).A
            pts_R = self._transform_gripper_to_world(mesh_pts, T_base_link_R, self.T_front2base_right)
            all_pts.append(pts_R)
            all_is_gripper.append(np.zeros(len(pts_R), dtype=bool))

        # gripper links (link7/link8)
        for ets, mesh_pts in [(self.ets_link7, self.mesh_link7_pts),
                              (self.ets_link8, self.mesh_link8_pts)]:
            T_L = ets.fkine(q_left).A
            pts_L = self._transform_gripper_to_world(mesh_pts, T_L, self.T_front2base_left)
            all_pts.append(pts_L)
            all_is_gripper.append(np.ones(len(pts_L), dtype=bool))

            T_R = ets.fkine(q_right).A
            pts_R = self._transform_gripper_to_world(mesh_pts, T_R, self.T_front2base_right)
            all_pts.append(pts_R)
            all_is_gripper.append(np.ones(len(pts_R), dtype=bool))

        return np.concatenate(all_pts, axis=0), np.concatenate(all_is_gripper, axis=0)

    def _render_robot_mask_from_points(
            self,
            pts_world: np.ndarray,
            W2C: np.ndarray,
            K_intrinsic: np.ndarray,
            output_size: Tuple[int, int],
    ) -> np.ndarray:
        """
        将 FK 机械臂点云投影到 2D 图像平面，通过形态学操作生成实心 mask。

        Pipeline: 投影 → 单像素填充 → dilate (扩散) → morphological close (填补缝隙)
        注意: 此处不做边界膨胀，边界膨胀由下游 _enhance_robot_mask 统一处理。

        Returns:
            mask: [H, W] bool
        """
        H, W = output_size
        mask = np.zeros((H, W), dtype=np.uint8)

        uv, valid = self._project_points_to_2d(pts_world, W2C, K_intrinsic, output_size)
        if valid.sum() == 0:
            return mask.astype(bool)

        u_px = np.round(uv[valid, 0]).astype(int)
        v_px = np.round(uv[valid, 1]).astype(int)
        mask[v_px, u_px] = 255

        # dilate: 扩散单像素标记为小区域
        kernel_init = np.ones((7, 7), np.uint8)
        mask = cv2.dilate(mask, kernel_init, iterations=1)

        # morphological close: 填补点云投影间的缝隙
        kernel_close = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=3)

        return mask.astype(bool)

    def _prepare_robot_data_from_fk(
            self,
            current_future_action: np.ndarray,
            output_size: Tuple[int, int],
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[List[np.ndarray]],
               List[List[np.ndarray]], Dict[str, np.ndarray]]:
        """
        通过 FK+URDF+mesh 计算 depth_point2double_cond 需要的 5 项数据，
        替代 simulator 提供的输入。

        Returns:
            points: K+1 帧 robot pointcloud, each [N_robot, 3]
            pointmask: K+1 帧 gripper bool mask, each [N_robot]
            rendermask: (K+1) x 3 视角 bool mask
            renderpose: (K+1) x 3 视角 4x4 W2C 外参
            intrinsics: 三视角内参 (缩放到 output_size)
        """
        K = len(current_future_action) - 1
        rgb_hw = (480, 640)

        output_intrinsics = {}
        for cam in self.camera_names:
            output_intrinsics[cam] = self._scale_intrinsics(
                self.intrinsics[cam], rgb_hw, output_size
            )

        points_list = []
        pointmask_list = []
        rendermask_list = []
        renderpose_list = []

        _silent_print(f"\n🤖 FK 生成 robot 数据 (K+1={K + 1} 帧, output_size={output_size})")
        for t in range(K + 1):
            pts, gripper_mask = self._compute_full_robot_points(current_future_action[t])
            extr = self.forward_kinematics(current_future_action[t])

            points_list.append(pts)
            pointmask_list.append(gripper_mask)

            frame_masks = []
            frame_poses = []
            for cam in self.camera_names:
                mask = self._render_robot_mask_from_points(
                    pts, extr[cam], output_intrinsics[cam], output_size
                )
                frame_masks.append(mask)
                frame_poses.append(extr[cam])
            rendermask_list.append(frame_masks)
            renderpose_list.append(frame_poses)

        _silent_print(f"  每帧 robot 点数: {len(points_list[0])}, "
              f"gripper 点数: {pointmask_list[0].sum()}")

        return points_list, pointmask_list, rendermask_list, renderpose_list, output_intrinsics

    def get_cond_gripper_interact(
            self,
            DA3_pts: Tuple[np.ndarray, np.ndarray, np.ndarray],
            gripper_pts: Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]],
            extrinsics_list: List[Dict[str, np.ndarray]],
    ) -> List[List[np.ndarray]]:
        """
        生成gripper交互条件（3D碰撞检测 + 投影到三视角）
        每个gripper独立检测碰撞并染色。

        纯 FK 版本: 所有相机（含 wrist）均通过 FK W2C 外参投影。
        """
        scene_pts = np.concatenate([DA3_pts[0], DA3_pts[1], DA3_pts[2]], axis=0)
        scene_xyz = scene_pts[:, :3]
        kdtree = KDTree(scene_xyz) if len(scene_xyz) > 0 else None

        lg1_list, lg2_list, rg1_list, rg2_list = gripper_pts
        K = len(lg1_list) - 1

        color_collision = np.array([1.0, 0.0, 0.0])
        color_free = np.array([0.5, 0.5, 0.5])

        def colorize(pts, collided):
            color = color_collision if collided else color_free
            return np.concatenate([pts, np.tile(color, (len(pts), 1))], axis=-1)

        cond_video = []

        _silent_print(f"🎯 生成gripper交互条件 (K={K} 时间步, 4个gripper独立检测)")

        for t in range(1, K + 1):
            collision_lg1 = self._check_collision(lg1_list[t], kdtree)
            collision_lg2 = self._check_collision(lg2_list[t], kdtree)
            collision_rg1 = self._check_collision(rg1_list[t], kdtree)
            collision_rg2 = self._check_collision(rg2_list[t], kdtree)

            all_grippers = np.concatenate([
                colorize(lg1_list[t], collision_lg1),
                colorize(lg2_list[t], collision_lg2),
                colorize(rg1_list[t], collision_rg1),
                colorize(rg2_list[t], collision_rg2),
            ], axis=0)

            extrinsics_t = extrinsics_list[t]

            frame_images = []
            for cam in self.camera_names:
                extrinsic = extrinsics_t[cam]
                proj_image = self._project_gripper_to_image(
                    gripper_points=all_grippers,
                    cam_name=cam,
                    extrinsic=extrinsic
                )
                frame_images.append(proj_image)

            cond_video.append(frame_images)

            _silent_print(f"  时刻 T+{t}: "
                  f"左G1={'碰撞' if collision_lg1 else '自由'}, "
                  f"左G2={'碰撞' if collision_lg2 else '自由'}, "
                  f"右G1={'碰撞' if collision_rg1 else '自由'}, "
                  f"右G2={'碰撞' if collision_rg2 else '自由'}")

        return cond_video

    def _check_collision(
            self,
            gripper_points: np.ndarray,
            scene_kdtree: Optional[KDTree]
    ) -> bool:
        if scene_kdtree is None or len(gripper_points) == 0:
            return False

        neighbors = scene_kdtree.query_ball_point(
            gripper_points,
            r=self.collision_threshold
        )

        # 每个gripper点周围的scene点数量
        neighbor_counts = np.array([len(n) for n in neighbors])

        # 至少有若干gripper点周围scene点密集
        contact_points = neighbor_counts >= self.scene_density_threshold

        return contact_points.sum() >= self.gripper_contact_min_points

    def _project_gripper_to_image(
            self,
            gripper_points: np.ndarray,
            cam_name: str,
            extrinsic: np.ndarray,
            image_size: Tuple[int, int] = (480, 640)
    ) -> np.ndarray:
        """
        将gripper点云投影到相机图像
        """
        H, W = image_size
        intrinsic = self.intrinsics[cam_name]

        # 补全外参
        if extrinsic.shape == (3, 4):
            extrinsic = np.vstack([extrinsic, [0, 0, 0, 1]])

        # 转换到相机坐标系
        xyz_world = gripper_points[:, :3]
        colors = gripper_points[:, 3:6]

        xyz_homo = np.concatenate([xyz_world, np.ones((len(xyz_world), 1))], axis=-1)
        xyz_cam = (extrinsic @ xyz_homo.T).T[:, :3]

        # 过滤相机后面的点
        valid_mask = xyz_cam[:, 2] > 0.01
        if valid_mask.sum() == 0:
            return np.zeros((H, W, 3), dtype=np.uint8)

        xyz_cam = xyz_cam[valid_mask]
        colors = colors[valid_mask]

        # 投影
        X, Y, Z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
        u = intrinsic[0, 0] * (X / Z) + intrinsic[0, 2]
        v = intrinsic[1, 1] * (Y / Z) + intrinsic[1, 2]

        u_px = np.round(u).astype(int)
        v_px = np.round(v).astype(int)

        # 过滤图像范围内的点
        in_image = (u_px >= 0) & (u_px < W) & (v_px >= 0) & (v_px < H)
        u_px = u_px[in_image]
        v_px = v_px[in_image]
        colors = colors[in_image]

        # 创建投影图像
        proj_image = np.zeros((H, W, 3), dtype=np.uint8)
        proj_image[v_px, u_px] = (colors * 255).astype(np.uint8)

        # 膨胀操作使投影更明显
        kernel = np.ones((3, 3), np.uint8)
        proj_image = cv2.dilate(proj_image, kernel, iterations=2)

        return proj_image

    def load_scene_flow_model(
            self,
            checkpoint_path: Optional[str] = None,
            config_path: Optional[str] = None,
            K: int = 15,
    ):
        """
        加载 scene flow 预测模型。

        Args:
            checkpoint_path: 模型权重路径。None 时使用 placeholder（输出零 flow）。
            config_path: YAML 配置文件路径。提供时从中读取所有超参数
                （数据处理、模型结构、推理渲染）。
            K: 模型的 K_max（FlowHead 输出步数）。
               仅在无 config_path 时作为 fallback。
        """
        from flow_model.scene_flow_model import SceneFlowModel
        self.scene_flow_model = SceneFlowModel(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            device=str(self.device),
            K=K,
        )
        _silent_print(f"✓ SceneFlowModel 加载完成 (K_max={self.scene_flow_model.K})")

    def get_cond_scene_flow(
            self,
            DA3_pts: Tuple[np.ndarray, np.ndarray, np.ndarray],
            gripper_pts: Tuple[List[np.ndarray], List[np.ndarray],
                               List[np.ndarray], List[np.ndarray]],
            extrinsics_list: List[Dict[str, np.ndarray]],
            current_future_action: np.ndarray,
            arm_masks: Optional[Dict[str, np.ndarray]] = None,
            current_obs: Optional[List[np.ndarray]] = None,
    ) -> List[List[np.ndarray]]:
        """
        场景 3D flow 预测 → 投影为三视角 2D flow condition

        渲染参数（flow_vis_mode, render_config, include_ego_motion）均从
        SceneFlowModel 的 YAML config 读取。

        Args:
            DA3_pts: (front_pts, left_pts, right_pts)，每个 [N_i, 6] (xyz+rgb)
            gripper_pts: (lg1_list, lg2_list, rg1_list, rg2_list)
                每个为 List[np.ndarray]，长度 K+1，每个 [5000, 3]
            extrinsics_list: K+1 帧三视角外参
            current_future_action: [K+1, 14] qpos 序列
            arm_masks: 机械臂 mask 字典，用于在投影后遮盖机械臂区域

        Returns:
            cond_video: K 帧 x 3 视角 flow 图像 List[List[np.ndarray]]
        """
        if self.scene_flow_model is None:
            self.load_scene_flow_model(checkpoint_path=None, K=len(extrinsics_list) - 1)

        K = len(extrinsics_list) - 1

        # 1. 模型预测 3D flow（内部做场景点/gripper 采样，与训练一致）
        _silent_print(f"  🔮 调用 SceneFlowModel 预测 3D flow (K={K})")
        # 构建 DINOv2 所需的内参/外参数组
        ixt_array = np.stack([self.intrinsics[c] for c in self.camera_names])  # [3, 3, 3]
        ext_t0 = np.stack([extrinsics_list[0][c] for c in self.camera_names])  # [3, 4, 4]

        pred_3d_flow, scene_xyz, source_views = self.scene_flow_model.predict(
            scene_pts_tuple=DA3_pts,
            gripper_pts_tuple=gripper_pts,
            action=current_future_action,
            K=K,
            current_obs=current_obs,
            intrinsics=ixt_array,
            extrinsics_t0=ext_t0,
        )  # pred_3d_flow: [N_sampled, K, 3], scene_xyz: [N_sampled, 3], source_views: [N_sampled]
        flow_mag = np.linalg.norm(pred_3d_flow, axis=-1)  # [N, K]
        _silent_print(f"  ✓ 3D flow 预测完成: shape={pred_3d_flow.shape}")
        _silent_print(f"    📊 3D flow 幅度: mean={flow_mag.mean():.4f}, max={flow_mag.max():.4f}, "
              f"nonzero={np.count_nonzero(flow_mag)}/{flow_mag.size}")

        # 模型可能截断 K，以实际输出为准
        K = pred_3d_flow.shape[1]

        # source_view 投影：每个点只投回来源视角（与 source_view 训练一致）
        sv = source_views if self.scene_flow_model.source_view_projection else None

        # 2. 投影为三视角 2D flow 图（参数全部从 YAML config 读取）
        cond_video = self._project_scene_flow_to_views(
            scene_xyz=scene_xyz,
            pred_3d_flow=pred_3d_flow,
            extrinsics_list=extrinsics_list[:K + 1],
            K=K,
            arm_masks=arm_masks,
            flow_vis_mode=self.scene_flow_model.flow_vis_mode,
            render_config=self.scene_flow_model.render_config,
            include_ego_motion=self.scene_flow_model.include_ego_motion,
            source_views=sv,
        )

        return cond_video

    def _project_scene_flow_to_views(
            self,
            scene_xyz: np.ndarray,
            pred_3d_flow: np.ndarray,
            extrinsics_list: List[Dict[str, np.ndarray]],
            K: int,
            image_size: Tuple[int, int] = (480, 640),
            arm_masks: Optional[Dict[str, np.ndarray]] = None,
            flow_vis_mode: str = "hsv",
            render_config: Optional[Dict] = None,
            include_ego_motion: bool = True,
            source_views: Optional[np.ndarray] = None,
    ) -> List[List[np.ndarray]]:
        """
        将 3D flow 投影到三视角 2D，渲染为 flow 图像。

        投影逻辑（per frame k, per camera）:
            1. 投影 scene_xyz(T=0) → (u0, v0) 使用第 0 帧外参
            2. future_xyz = scene_xyz + pred_flow[:, k]
            3. include_ego_motion=True:  投影 future_xyz → (uk, vk) 使用第 k+1 帧外参
               include_ego_motion=False: 投影 future_xyz → (uk, vk) 使用第 0 帧外参
            4. 2D flow = (uk - u0, vk - v0)
            5. 用 arm_masks 遮盖机械臂区域

        Args:
            scene_xyz: [N, 3] 场景点 T=0 坐标
            pred_3d_flow: [N, K, 3]
            extrinsics_list: K+1 帧外参
            K: 预测步数
            image_size: (H, W)
            arm_masks: 机械臂 mask 字典，key 为 camera name
            flow_vis_mode: "hsv", "arrow", 或 "xy_mask"
            render_config: 渲染超参数（见 get_cond_scene_flow docstring）
            include_ego_motion: True 时 future 点用 ext_k 投影（含 ego-motion），
                               False 时统一用 ext_0 投影（纯场景 flow）
            source_views: [N] int32, 每个场景点的来源视角 (0=front, 1=left, 2=right)。
                         提供时每个点只投回来源视角（source_view 监督模式，与训练一致）；
                         None 时所有点投到所有视角（multi_view/front_main/front_only 模式）。

        Returns:
            cond_video: K x 3 flow images
        """
        rc = render_config or {}
        dilate_kernel = rc.get("dilate_kernel", 3)
        dilate_iter = rc.get("dilate_iter", 2)
        min_val = rc.get("min_val", 30)
        flow_scale = rc.get("flow_scale", 50.0)
        max_points = rc.get("max_points", 2000)

        cond_video = []

        for k in range(K):
            frame_images = []
            future_xyz = scene_xyz + pred_3d_flow[:, k, :]  # [N, 3]

            for cam_idx, cam in enumerate(self.camera_names):
                intrinsic = self.intrinsics[cam]
                ext_0 = extrinsics_list[0][cam]
                ext_future = extrinsics_list[k + 1][cam] if include_ego_motion else ext_0

                # source_view 模式：只投该视角来源的点
                if source_views is not None:
                    sv_mask = source_views == cam_idx
                    cam_xyz = scene_xyz[sv_mask]
                    cam_future = future_xyz[sv_mask]
                else:
                    cam_xyz = scene_xyz
                    cam_future = future_xyz

                # 投影 T=0 点
                uv0, valid0 = self._project_points_to_2d(
                    cam_xyz, ext_0, intrinsic, image_size
                )
                # 投影 future 点
                uvk, validk = self._project_points_to_2d(
                    cam_future, ext_future, intrinsic, image_size
                )

                # 仅保留两次投影都有效的点
                both_valid = valid0 & validk
                if both_valid.sum() == 0:
                    frame_images.append(np.zeros((*image_size, 3), dtype=np.uint8))
                    continue

                flow_2d = uvk[both_valid] - uv0[both_valid]  # [M, 2]
                anchor_uv = uv0[both_valid]  # [M, 2]

                # 与训练 GT 一致：投影后 2D flow 幅度 < 阈值的置零（训练时该范围无监督）
                if self.scene_flow_model is not None and self.scene_flow_model.gt_flow_threshold > 0:
                    flow_mag_2d = np.linalg.norm(flow_2d, axis=1)
                    flow_2d[flow_mag_2d < self.scene_flow_model.gt_flow_threshold] = 0.0

                if flow_vis_mode == "arrow":
                    vis_image = self._render_flow_arrows(
                        anchor_uv, flow_2d, image_size, max_points=max_points,
                    )
                elif flow_vis_mode == "xy_mask":
                    vis_image = self._render_flow_xy_mask(
                        anchor_uv, flow_2d, image_size,
                        dilate_kernel=dilate_kernel, dilate_iter=dilate_iter,
                        flow_scale=flow_scale,
                    )
                else:
                    vis_image = self._render_flow_hsv(
                        anchor_uv, flow_2d, image_size,
                        dilate_kernel=dilate_kernel, dilate_iter=dilate_iter,
                        min_val=min_val,
                    )

                # 用 arm_masks 遮盖机械臂区域
                if arm_masks is not None and cam in arm_masks:
                    vis_image[arm_masks[cam]] = 0

                frame_images.append(vis_image)

            cond_video.append(frame_images)

            if (k + 1) % max(1, K // 5) == 0 or k == K - 1:
                _silent_print(f"  🎨 scene flow → {flow_vis_mode.upper()}: 帧 {k + 1}/{K} 完成")

        return cond_video

    @staticmethod
    def _project_points_to_2d(
            xyz_world: np.ndarray,
            extrinsic: np.ndarray,
            intrinsic: np.ndarray,
            image_size: Tuple[int, int] = (480, 640),
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        3D 世界坐标 → 2D 像素坐标。

        Args:
            xyz_world: [N, 3]
            extrinsic: [4, 4] or [3, 4] W2C 外参
            intrinsic: [3, 3] 内参
            image_size: (H, W)

        Returns:
            uv: [N, 2] float (u, v)
            valid: [N] bool
        """
        H, W = image_size
        N = xyz_world.shape[0]

        # 补全外参
        if extrinsic.shape == (3, 4):
            extrinsic = np.vstack([extrinsic, [0, 0, 0, 1]])

        # 世界 → 相机
        xyz_homo = np.concatenate([xyz_world, np.ones((N, 1))], axis=-1)  # [N, 4]
        xyz_cam = (extrinsic @ xyz_homo.T).T[:, :3]  # [N, 3]

        z = xyz_cam[:, 2]
        valid_z = z > 0.01

        # 投影
        uv = np.zeros((N, 2), dtype=np.float64)
        if valid_z.any():
            X = xyz_cam[valid_z, 0]
            Y = xyz_cam[valid_z, 1]
            Z = xyz_cam[valid_z, 2]
            u = intrinsic[0, 0] * (X / Z) + intrinsic[0, 2]
            v = intrinsic[1, 1] * (Y / Z) + intrinsic[1, 2]
            uv[valid_z, 0] = u
            uv[valid_z, 1] = v

        # 图像范围内
        u_int = np.round(uv[:, 0]).astype(int)
        v_int = np.round(uv[:, 1]).astype(int)
        in_image = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H)

        valid = valid_z & in_image
        return uv, valid

    @staticmethod
    def _render_flow_hsv(
            anchor_uv: np.ndarray,
            flow_2d: np.ndarray,
            image_size: Tuple[int, int] = (480, 640),
            dilate_kernel: int = 3,
            dilate_iter: int = 2,
            min_val: int = 30,
    ) -> np.ndarray:
        """
        稀疏 2D flow → 稠密 HSV 图像。

        HSV 编码:
            Hue = flow 方向 (0-180, OpenCV HSV)
            Saturation = 255
            Value = flow 幅度 (99th percentile 归一化到 min_val-255)
                    所有有投影的场景点至少有 min_val 亮度，
                    使 world model 能区分"有场景点但不动"和"无数据背景"。

        Args:
            anchor_uv: [M, 2] float 锚点像素坐标
            flow_2d: [M, 2] float 2D flow
            image_size: (H, W)
            dilate_kernel: 膨胀核大小
            dilate_iter: 膨胀迭代次数
            min_val: 静止点的最低亮度 (0-255)，使场景覆盖可见

        Returns:
            hsv_bgr: [H, W, 3] uint8 BGR 图像（从 HSV 转换）
        """
        H, W = image_size

        # flow 幅度和方向
        mag = np.linalg.norm(flow_2d, axis=1)  # [M]
        angle = np.arctan2(flow_2d[:, 1], flow_2d[:, 0])  # [M], radians

        # 归一化幅度（99th percentile 避免 outlier）
        if mag.max() > 1e-6:
            mag_cap = np.percentile(mag, 99)
            mag_norm = np.clip(mag / max(mag_cap, 1e-6), 0.0, 1.0)
        else:
            mag_norm = np.zeros_like(mag)

        # HSV 图像
        hsv = np.zeros((H, W, 3), dtype=np.uint8)

        # splat 到最近像素
        u_px = np.round(anchor_uv[:, 0]).astype(int)
        v_px = np.round(anchor_uv[:, 1]).astype(int)
        in_img = (u_px >= 0) & (u_px < W) & (v_px >= 0) & (v_px < H)

        if in_img.sum() > 0:
            u_px = u_px[in_img]
            v_px = v_px[in_img]
            angle_valid = angle[in_img]
            mag_valid = mag_norm[in_img]

            # Hue: 角度 [0, 2*pi] → [0, 180] (OpenCV HSV hue range)
            hue = ((angle_valid % (2 * np.pi)) / (2 * np.pi) * 180).astype(np.uint8)
            sat = np.full_like(hue, 255)
            # Value: min_val ~ 255，静止点也有最低亮度
            val = (min_val + mag_valid * (255 - min_val)).astype(np.uint8)

            hsv[v_px, u_px, 0] = hue
            hsv[v_px, u_px, 1] = sat
            hsv[v_px, u_px, 2] = val

        # 膨胀填充
        if dilate_kernel > 0 and dilate_iter > 0:
            kernel = np.ones((dilate_kernel, dilate_kernel), np.uint8)
            # 只膨胀有值的区域（value > 0）
            mask = (hsv[:, :, 2] > 0).astype(np.uint8)
            mask_dilated = cv2.dilate(mask, kernel, iterations=dilate_iter)
            # 对 HSV 三通道分别膨胀
            for c in range(3):
                hsv[:, :, c] = cv2.dilate(hsv[:, :, c], kernel, iterations=dilate_iter)
            # 限制在膨胀 mask 内
            hsv[mask_dilated == 0] = 0

        # HSV → BGR
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return bgr

    @staticmethod
    def _render_flow_xy_mask(
            anchor_uv: np.ndarray,
            flow_2d: np.ndarray,
            image_size: Tuple[int, int] = (480, 640),
            dilate_kernel: int = 3,
            dilate_iter: int = 2,
            flow_scale: float = 50.0,
    ) -> np.ndarray:
        """
        稀疏 2D flow → 3 通道 (flow_x, flow_y, mask) 图像。

        神经网络友好编码，无 HSV 非线性转换：
            Ch0 = flow_x 归一化到 [0, 255]（128 为零点）
            Ch1 = flow_y 归一化到 [0, 255]（128 为零点）
            Ch2 = mask（有投影=255, 无投影=0）

        Args:
            anchor_uv: [M, 2] float 锚点像素坐标
            flow_2d: [M, 2] float 2D flow
            image_size: (H, W)
            dilate_kernel: 膨胀核大小
            dilate_iter: 膨胀迭代次数
            flow_scale: flow 归一化尺度（px），[-flow_scale, +flow_scale] → [0, 255]

        Returns:
            xy_mask: [H, W, 3] uint8 图像
        """
        H, W = image_size
        result = np.zeros((H, W, 3), dtype=np.uint8)

        u_px = np.round(anchor_uv[:, 0]).astype(int)
        v_px = np.round(anchor_uv[:, 1]).astype(int)
        in_img = (u_px >= 0) & (u_px < W) & (v_px >= 0) & (v_px < H)

        if in_img.sum() > 0:
            u_px = u_px[in_img]
            v_px = v_px[in_img]
            fx = flow_2d[in_img, 0]
            fy = flow_2d[in_img, 1]

            # 归一化: [-flow_scale, +flow_scale] → [0, 255]，128 为零点
            ch0 = np.clip(fx / flow_scale * 127.0 + 128.0, 0, 255).astype(np.uint8)
            ch1 = np.clip(fy / flow_scale * 127.0 + 128.0, 0, 255).astype(np.uint8)

            result[v_px, u_px, 0] = ch0
            result[v_px, u_px, 1] = ch1
            result[v_px, u_px, 2] = 255  # mask

        # 膨胀填充
        if dilate_kernel > 0 and dilate_iter > 0:
            kernel = np.ones((dilate_kernel, dilate_kernel), np.uint8)
            mask = (result[:, :, 2] > 0).astype(np.uint8)
            mask_dilated = cv2.dilate(mask, kernel, iterations=dilate_iter)
            for c in range(3):
                result[:, :, c] = cv2.dilate(result[:, :, c], kernel, iterations=dilate_iter)
            result[mask_dilated == 0] = 0

        return result

    @staticmethod
    def _render_flow_arrows(
            anchor_uv: np.ndarray,
            flow_2d: np.ndarray,
            image_size: Tuple[int, int] = (480, 640),
            bg_image: Optional[np.ndarray] = None,
            max_points: int = 2000,
            min_arrow_px: float = 1.0,
    ) -> np.ndarray:
        """
        稀疏 2D flow → debug 可视化图像。

        所有投影点都会显示（静止点画圆点，运动点画箭头），
        便于观察参与预测的场景点在各视角的分布。
        超过 max_points 时均匀采样保留全局分布。

        Args:
            anchor_uv: [M, 2] float 锚点像素坐标
            flow_2d: [M, 2] float 2D flow
            image_size: (H, W)
            bg_image: 可选 BGR 底图，None 则用黑底
            max_points: 最大可视化点数，超出时均匀采样
            min_arrow_px: 幅度 >= 此值(px)的点画箭头，< 此值画圆点

        Returns:
            canvas: [H, W, 3] uint8 BGR 图像
        """
        H, W = image_size

        # 底图
        if bg_image is not None:
            canvas = bg_image.copy()
            if canvas.shape[:2] != (H, W):
                canvas = cv2.resize(canvas, (W, H))
        else:
            canvas = np.zeros((H, W, 3), dtype=np.uint8)

        M = anchor_uv.shape[0]
        if M == 0:
            return canvas

        mag = np.linalg.norm(flow_2d, axis=1)  # [M]

        # 超出 max_points 时均匀采样，保留全局分布
        if M > max_points:
            idx = np.linspace(0, M - 1, max_points, dtype=int)
            anchor_uv = anchor_uv[idx]
            flow_2d = flow_2d[idx]
            mag = mag[idx]
            M = max_points

        # 幅度归一化 → colormap 索引 [0, 255]
        mag_max = mag.max()
        if mag_max > 1e-6:
            mag_norm = np.clip(mag / mag_max, 0.0, 1.0)
        else:
            mag_norm = np.zeros_like(mag)

        # JET colormap: 蓝(静止/小flow) → 红(大flow)
        color_indices = (mag_norm * 255).astype(np.uint8)
        colormap = cv2.applyColorMap(
            color_indices.reshape(-1, 1), cv2.COLORMAP_JET
        ).reshape(-1, 3)  # [M, 3] BGR

        # 先画静止点（圆点），再画运动点（箭头），运动箭头覆盖在上层
        moving = mag >= min_arrow_px
        static = ~moving

        # 静止点: 灰色圆点
        for i in np.where(static)[0]:
            pt = (int(round(anchor_uv[i, 0])), int(round(anchor_uv[i, 1])))
            cv2.circle(canvas, pt, 2, (128, 128, 128), -1)

        # 运动点: 彩色箭头
        for i in np.where(moving)[0]:
            pt1 = (int(round(anchor_uv[i, 0])), int(round(anchor_uv[i, 1])))
            pt2 = (int(round(anchor_uv[i, 0] + flow_2d[i, 0])),
                   int(round(anchor_uv[i, 1] + flow_2d[i, 1])))
            color = tuple(int(c) for c in colormap[i])
            tip_length = min(0.3, 8.0 / max(mag[i], 1e-6))
            cv2.arrowedLine(canvas, pt1, pt2, color,
                thickness=1, tipLength=tip_length)

        return canvas

    def get_cond_implicit(self, DA3_pts, gripper_pts):
        """隐式特征条件（暂未实现）"""
        raise NotImplementedError("implicit_3D condition not implemented yet!")

    def rgb_action2flow_cond(
            self,
            current_obs: List[np.ndarray],
            current_future_action: np.ndarray,
            modality: str = "3D",
            cond: str = "gripper_interact",
    ) -> Tuple[List[List[np.ndarray]], Dict[str, np.ndarray]]:
        """
        从RGB图像和动作序列生成flow条件

        scene_flow 条件的渲染参数（flow_vis_mode, render_config, include_ego_motion）
        均从 SceneFlowModel 的 YAML config 读取，无需在此传入。

        Returns:
            cond_video: 条件视频
            arm_masks: 机械臂过滤mask字典
        """
        _silent_print(f"\n{'=' * 70}")
        _silent_print(f"  RGB + Action → Flow Condition (cond={cond})")
        _silent_print(f"{'=' * 70}\n")
        K = len(current_future_action) - 1

        # Step 1: 计算未来K帧的相机外参（用于投影）
        _silent_print(f"\nStep 1: 计算未来K帧相机外参")
        extrinsics_list = []
        for t in range(0, K + 1):
            ext_t = self.forward_kinematics(current_future_action[t])
            extrinsics_list.append(ext_t)
        _silent_print(f"  已计算 {len(extrinsics_list)} 帧外参")

        # Step 2: DA3推理获取第T帧的深度
        _silent_print(f"\nStep 2: DA3深度估计")
        depths, extrinsics_da3, intrinsics = self.forward_DA3(current_obs, extrinsics_list[0])

        if modality == "3D":
            # Step 3: 深度转点云（过滤机械臂）
            _silent_print(f"\nStep 3: 深度转点云（过滤机械臂）")
            front_pts, left_pts, right_pts, arm_masks = self.convert_depth(
                current_obs, depths, extrinsics_da3, intrinsics
            )

            # Step 4: 计算当前帧+未来K帧gripper点云（共K+1帧）
            _silent_print(f"\nStep 4: 计算当前帧+未来K帧gripper点云")
            gripper_pts_lg1, gripper_pts_lg2, gripper_pts_rg1, gripper_pts_rg2 = \
                self.get_gripper_points(current_future_action)
            _silent_print(f"  总帧数: K+1={K + 1} (当前帧+未来{K}帧)")

        if modality == "2D":
            # Step 3: 计算当前帧+未来K帧gripper点云（共K+1帧）
            _silent_print(f"\nStep 3: 计算当前帧+未来K帧gripper点云")
            gripper_pts_lg1, gripper_pts_lg2, gripper_pts_rg1, gripper_pts_rg2 = \
                self.get_gripper_points(current_future_action)  # (K+1)x5000x3
            _silent_print(f"  总帧数: K+1={K + 1} (当前帧+未来{K}帧)")

            # step 4: 获取当前帧gripper点云到未来帧的3D flow
            gripper_flow_lg1, gripper_flow_lg2, gripper_flow_rg1, gripper_flow_rg2 = \
                self.get_gripper_flow(gripper_pts_lg1, gripper_pts_lg2, gripper_pts_rg1, gripper_pts_rg2)  # 5000xKx3

            # step 5: 将3D flow投影到2D pixel（纯 FK W2C 投影）
            _silent_print(f"\nStep 5: 3D flow → 2D 投影 (FK W2C)")
            gripper_flow_lg_2D = self.get_flow_project_refine(
                gripper_flow_lg1, gripper_flow_lg2,
                gripper_pts_lg1[0], gripper_pts_lg2[0],
                extrinsics_list[0]['left'], cam_name='left')
            gripper_flow_rg_2D = self.get_flow_project_refine(
                gripper_flow_rg1, gripper_flow_rg2,
                gripper_pts_rg1[0], gripper_pts_rg2[0],
                extrinsics_list[0]['right'], cam_name='right')

            # Step 6: 深度转场景点云 + 重建K+1帧gripper点云
            # front基于颜色过滤，wrist基于flow mask区分场景/gripper
            # gripper用depth反投影+3D flow重建K+1帧点云（格式同3D方案）
            _silent_print(f"\nStep 6: 深度转场景点云 + 重建K+1帧gripper点云")
            front_pts, left_pts, right_pts, arm_masks, gripper_pts_lg1, gripper_pts_lg2, gripper_pts_rg1, gripper_pts_rg2 = \
                self.convert_depth_with_flow_mask(
                    current_obs, depths, extrinsics_da3, intrinsics,
                    gripper_flow_lg_2D, gripper_flow_rg_2D
                )

        # Step 5: 生成条件
        if cond == "gripper_interact":
            _silent_print(f"\nStep 5: 生成gripper交互条件")
            cond_video = self.get_cond_gripper_interact(
                (front_pts, left_pts, right_pts),
                (gripper_pts_lg1, gripper_pts_lg2, gripper_pts_rg1, gripper_pts_rg2),
                extrinsics_list=extrinsics_list,
            )
            return cond_video, arm_masks
        elif cond == "scene_flow":
            _silent_print(f"\nStep 5: 生成场景3D flow条件")
            cond_video = self.get_cond_scene_flow(
                DA3_pts=(front_pts, left_pts, right_pts),
                gripper_pts=(gripper_pts_lg1, gripper_pts_lg2, gripper_pts_rg1, gripper_pts_rg2),
                extrinsics_list=extrinsics_list,
                current_future_action=current_future_action,
                arm_masks=arm_masks,
                current_obs=current_obs,
            )
            return cond_video, arm_masks
        elif cond == "implicit":
            _silent_print(f"\nStep 5: 生成隐式3D条件")
            cond_feature = self.get_cond_implicit(
                (front_pts, left_pts, right_pts),
                (gripper_pts_lg1, gripper_pts_lg2, gripper_pts_rg1, gripper_pts_rg2)
            )
            return cond_feature, arm_masks
        else:
            raise NotImplementedError(f"{cond} not implemented yet!")

    # ==================== Double-stream condition helpers ====================

    def _depth_to_xyz(
            self,
            depth: np.ndarray,
            intrinsic: np.ndarray,
            mask_exclude: Optional[np.ndarray] = None,
            depth_range: Tuple[float, float] = (0.0, 5.0),
    ) -> np.ndarray:
        """
        深度图 → xyz 点云 (无颜色)

        Args:
            depth: [H, W] 深度图
            intrinsic: [3, 3] 内参（必须匹配 depth 分辨率）
            mask_exclude: [H, W] bool, True 的像素被排除（会自动 resize）
            depth_range: 有效深度范围

        Returns:
            points_xyz: [N, 3] 相机坐标系下的 3D 点
        """
        H, W = depth.shape

        if mask_exclude is not None and mask_exclude.shape[:2] != (H, W):
            mask_exclude = cv2.resize(
                mask_exclude.astype(np.uint8), (W, H),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        u, v = np.meshgrid(np.arange(W), np.arange(H))

        valid_mask = (depth > depth_range[0]) & (depth < depth_range[1])
        if mask_exclude is not None:
            valid_mask = valid_mask & (~mask_exclude)

        u_valid = u[valid_mask]
        v_valid = v[valid_mask]
        z_valid = depth[valid_mask].astype(np.float64)

        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]

        x = (u_valid - cx) * z_valid / fx
        y = (v_valid - cy) * z_valid / fy

        return np.stack([x, y, z_valid], axis=-1)

    @staticmethod
    def _scale_intrinsics(
            intrinsic: np.ndarray,
            src_hw: Tuple[int, int],
            dst_hw: Tuple[int, int],
    ) -> np.ndarray:
        """
        按分辨率比例缩放内参 (fx, fy, cx, cy)

        Args:
            intrinsic: [3, 3] 原始内参
            src_hw: (H_src, W_src) 原始分辨率
            dst_hw: (H_dst, W_dst) 目标分辨率

        Returns:
            scaled: [3, 3] 缩放后的内参
        """
        scale_h = dst_hw[0] / src_hw[0]
        scale_w = dst_hw[1] / src_hw[1]
        scaled = intrinsic.copy()
        scaled[0, 0] *= scale_w  # fx
        scaled[0, 2] *= scale_w  # cx
        scaled[1, 1] *= scale_h  # fy
        scaled[1, 2] *= scale_h  # cy
        return scaled

    @staticmethod
    def _transform_xyz_to_world(
            pts_cam: np.ndarray,
            extrinsic: np.ndarray,
    ) -> np.ndarray:
        """
        [N, 3] camera coords → world coords (C2W = inv(W2C))

        Args:
            pts_cam: [N, 3] 相机坐标系下的点
            extrinsic: [4, 4] or [3, 4] W2C 外参

        Returns:
            pts_world: [N, 3] 世界坐标系下的点
        """
        if extrinsic.shape == (3, 4):
            extrinsic = np.vstack([extrinsic, [0, 0, 0, 1]])
        C2W = np.linalg.inv(extrinsic)
        pts_homo = np.concatenate([pts_cam, np.ones((len(pts_cam), 1))], axis=-1)
        return (C2W @ pts_homo.T).T[:, :3]

    @staticmethod
    def _gpu_min_dist(
            src: torch.Tensor,
            tgt: torch.Tensor,
            chunk_size: int = 2048,
    ) -> torch.Tensor:
        """
        GPU 加速最近邻距离: 对 src 中每个点计算到 tgt 的最小欧式距离。
        分块计算避免 OOM。

        Args:
            src: [N, 3] float tensor on GPU
            tgt: [M, 3] float tensor on GPU
            chunk_size: 每次处理的 src 点数

        Returns:
            [N] float tensor, 最近邻距离
        """
        N = src.shape[0]
        M = tgt.shape[0]
        empty_target_fallback_dist = 3.0  # meter
        if N == 0:
            return torch.empty(0, device=src.device, dtype=src.dtype)
        if M == 0:
            # 与空点集最近邻距离不可定义，使用较大常量距离做平滑退化，
            # 避免 +inf 在后续统计/归一化中造成突变。
            return torch.full(
                (N,), empty_target_fallback_dist, device=src.device, dtype=src.dtype
            )
        min_dists = torch.empty(N, device=src.device)
        for i in range(0, N, chunk_size):
            dists = torch.cdist(src[i:i + chunk_size], tgt)  # [chunk, M]
            min_dists[i:i + chunk_size] = dists.min(dim=1).values
            del dists
        return min_dists

    @staticmethod
    def _sparse_to_dense_in_mask(
            sparse_map: np.ndarray,
            occ_mask: np.ndarray,
            fill_mask: np.ndarray,
    ) -> np.ndarray:
        """
        稀疏距离值 → EDT 最近邻插值填充 mask 区域

        Args:
            sparse_map: [H, W] float, 有值的像素位置存放距离
            occ_mask: [H, W] bool, sparse_map 中有值的像素位置
            fill_mask: [H, W] bool, 需要填充的目标区域

        Returns:
            dense_map: [H, W] float, fill_mask 内全部填满
        """
        dense_map = sparse_map.copy()
        if occ_mask.sum() == 0:
            return dense_map
        _, nearest_idx = distance_transform_edt(~occ_mask, return_indices=True)
        need_fill = fill_mask & ~occ_mask
        dense_map[need_fill] = sparse_map[
            nearest_idx[0][need_fill], nearest_idx[1][need_fill]
        ]
        return dense_map

    @staticmethod
    def _dist_to_heatmap(
            dist_map: np.ndarray,
            mask: np.ndarray,
            cap: Optional[float] = None,
    ) -> np.ndarray:
        """
        距离图 → heatmap 图像 (JET colormap)

        近距离 = 红(热), 远距离 = 蓝(冷), mask 外 = 黑色

        Args:
            dist_map: [H, W] float, 距离值（mask 内应已通过 EDT 填充）
            mask: [H, W] bool, 有效区域
            cap: 归一化上限。None 时用当前帧 99th percentile（独立归一化），
                 提供时用全局 cap（跨帧一致性）

        Returns:
            heatmap: [H, W, 3] uint8 RGB 图像
        """
        H, W = dist_map.shape
        heatmap = np.zeros((H, W, 3), dtype=np.uint8)

        if mask.sum() == 0:
            return heatmap

        vals = dist_map[mask]
        if not np.isfinite(vals).any():
            return heatmap

        if cap is None:
            finite_vals = vals[np.isfinite(vals)]
            cap = np.percentile(finite_vals, 99) if len(finite_vals) > 0 else 1.0
        cap = max(cap, 1e-6)

        norm = np.clip(dist_map / cap, 0.0, 1.0)

        # JET colormap: OpenCV 中 0=蓝, 255=红
        # 近距离(小 norm)应为红 → (1-norm)*255
        gray = ((1.0 - norm) * 255).astype(np.uint8)
        colored_bgr = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)

        heatmap[mask] = colored_rgb[mask]
        return heatmap

    # ==================== Double-stream condition main ====================

    def depth_point2double_cond(
            self,
            current_depth: List[np.ndarray],
            current_future_action: np.ndarray,
            current_future_points_robosim: Optional[List[np.ndarray]] = None,
            current_future_pointmask_robosim: Optional[List[np.ndarray]] = None,
            current_future_rendermask_robosim: Optional[List[List[np.ndarray]]] = None,
            current_future_renderpose_robosim: Optional[List[List[np.ndarray]]] = None,
            sim_intrinsics: Optional[Dict[str, np.ndarray]] = None,
            current_obs: Optional[List[np.ndarray]] = None,
            output_size: Optional[Tuple[int, int]] = None,
            use_gpu: bool = True,
    ) -> Tuple[List[List[np.ndarray]], List[List[np.ndarray]], Optional[Dict[str, np.ndarray]]]:
        """
        从 robot pointcloud、三视角 robot render mask、当前帧深度和动作得到
        two-stream cond videos。

        sim 五项输入 (points/pointmask/rendermask/renderpose/intrinsics) 可选:
        - 提供时使用 simulator 精确数据
        - 不提供时通过 FK+URDF+mesh 自动生成 (rendermask 用点云投影+形态学)

        cond_video1 (robot-centric):
            未来每帧 robot 每个点到场景的最近距离 → pose 投影到 robot mask 区域 → heatmap
        cond_video2 (scene-centric):
            场景每个点到未来每帧 robot 的最近距离 → FK pose 投影到 scene 区域 → heatmap

        Pipeline:
        1. 用 T=0 rendermask 从 depth 去除 robot → 三视角 FK 拼接得纯场景 pointcloud
        2. 逐帧计算双向距离 (GPU torch.cdist 或 CPU scipy KDTree, 由 use_gpu 控制):
           - robot→scene: 最近邻距离 + pose 投影 + EDT 插值到 rendermask
           - scene→robot: 最近邻距离 + FK pose 投影 + EDT 插值到 scene mask
        3. 距离值转 heatmap (placeholder: JET colormap)

        Args:
            current_depth: 当前帧三视角深度图 [front, left, right]
            current_future_action: [K+1, 14]，用 T=0 算 FK 外参
            current_future_points_robosim: (可选) K+1 帧 robot pointcloud，每个 [N_robot, 3]，
                front camera 坐标系。不提供时由 FK+URDF+mesh 生成。
            current_future_pointmask_robosim: (可选) K+1 帧 gripper bool mask，每个 [N_robot]，
                True 表示该点属于 gripper (link7/link8)
            current_future_rendermask_robosim: (可选) (K+1) x 3 视角 bool mask
                [t][cam_idx] → [H_mask, W_mask] bool
            current_future_renderpose_robosim: (可选) (K+1) x 3 视角 4x4 W2C 外参
                [t][cam_idx] → [4, 4]
            sim_intrinsics: (可选) 三视角内参
                {'front': [3,3], 'left': [3,3], 'right': [3,3]}
            current_obs: 可选三视角 RGB [front, left, right]，
                用于检查 T=0 robot mask 是否干净（输出叠加可视化）
            output_size: (可选) 输出分辨率 (H, W)，仅在 FK 模式下使用
                默认 (480, 640)，sim 模式自动取 rendermask 分辨率
            use_gpu: 最近邻距离计算是否使用 GPU (torch.cdist)。
                True (默认): 更快但占用显存; False: 使用 scipy KDTree (CPU)。

        Returns:
            cond_video1: K x 3 视角 heatmap (robot-centric)，分辨率 = output size
            cond_video2: K x 3 视角 heatmap (scene-centric)，分辨率 = output size
            arm_mask_debug: 当 current_obs 提供时，Dict[str, np.ndarray]
                三视角 RGB 上叠加 T=0 robot mask（绿色标注被 mask 区域）;
                否则 None
        """
        # === FK fallback: 若无 sim 数据，通过 FK+URDF+mesh 生成 ===
        if current_future_points_robosim is None:
            _output_size = output_size or (480, 640)
            (current_future_points_robosim, current_future_pointmask_robosim,
             current_future_rendermask_robosim, current_future_renderpose_robosim,
             sim_intrinsics) = self._prepare_robot_data_from_fk(
                current_future_action, _output_size
            )

        # === Step 0: 参数准备 ===
        K = len(current_future_points_robosim) - 1
        output_size = current_future_rendermask_robosim[0][0].shape[:2]  # (H_out, W_out)
        H_out, W_out = output_size

        # T=0 FK 外参 (用于 cond_video2 场景投影)
        fk_ext_t0 = self.forward_kinematics(current_future_action[0])

        # Depth intrinsics: RGB 内参缩放到 depth 分辨率
        rgb_hw = (480, 640)
        depth_intrinsics = {}
        for cam_idx, cam in enumerate(self.camera_names):
            depth_hw = current_depth[cam_idx].shape[:2]
            depth_intrinsics[cam] = self._scale_intrinsics(
                self.intrinsics[cam], rgb_hw, depth_hw
            )

        # 输出 intrinsics: RGB 内参缩放到 output (rendermask) 分辨率
        # (cond_video2 场景点投影用, 与 rendermask 分辨率对齐)
        output_intrinsics = {}
        for cam in self.camera_names:
            output_intrinsics[cam] = self._scale_intrinsics(
                self.intrinsics[cam], rgb_hw, output_size
            )

        _silent_print(f"\n{'=' * 70}")
        _silent_print(f"  depth_point2double_cond: K={K}, output_size={output_size}")
        _silent_print(f"{'=' * 70}\n")

        # === Step 1: 增强 T=0 robot mask (膨胀 + 颜色检测) ===
        enhanced_masks_t0 = {}  # cam → bool mask (rendermask 原始分辨率)
        arm_mask_debug = None
        for cam_idx, cam in enumerate(self.camera_names):
            raw_mask = current_future_rendermask_robosim[0][cam_idx]
            rgb = current_obs[cam_idx] if current_obs is not None else None
            enhanced = self._enhance_robot_mask(raw_mask, rgb=rgb)
            # _enhance_robot_mask 有 rgb 时返回 RGB 分辨率, 否则 rendermask 分辨率
            # 统一存 rendermask 分辨率版本（用于 scene mask / 点云排除）
            if rgb is not None and enhanced.shape[:2] != raw_mask.shape[:2]:
                H_rm, W_rm = raw_mask.shape[:2]
                enhanced_rm = cv2.resize(
                    enhanced.astype(np.uint8), (W_rm, H_rm),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            else:
                enhanced_rm = enhanced
            enhanced_masks_t0[cam] = enhanced_rm

        # arm_mask_debug: 在 RGB 上叠加 enhanced mask（半透明绿）
        if current_obs is not None:
            arm_mask_debug = {}
            for cam_idx, cam in enumerate(self.camera_names):
                rgb = current_obs[cam_idx]
                raw_mask = current_future_rendermask_robosim[0][cam_idx]

                # enhanced mask at RGB 分辨率
                mask_enhanced = self._enhance_robot_mask(raw_mask, rgb=rgb)
                # 原始 rendermask at RGB 分辨率
                H_rgb, W_rgb = rgb.shape[:2]
                if raw_mask.shape[:2] != (H_rgb, W_rgb):
                    mask_raw_rgb = cv2.resize(
                        raw_mask.astype(np.uint8), (W_rgb, H_rgb),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                else:
                    mask_raw_rgb = raw_mask.astype(bool)

                # 可视化: 原始 rendermask 区域=绿, 增强新增区域=黄, 其余=原 RGB
                overlay = rgb.astype(np.float32)
                # 绿色: 原始 rendermask 区域 (sim 直接覆盖)
                overlay[mask_raw_rgb] = overlay[mask_raw_rgb] * 0.5 + np.array([0, 255, 0]) * 0.5
                # 黄色: 增强新增区域 (膨胀 + 颜色检测补充)
                new_region = mask_enhanced & ~mask_raw_rgb
                overlay[new_region] = overlay[new_region] * 0.5 + np.array([255, 255, 0]) * 0.5
                arm_mask_debug[cam] = overlay.clip(0, 255).astype(np.uint8)

                coverage_raw = mask_raw_rgb.sum() / mask_raw_rgb.size * 100
                coverage_enh = mask_enhanced.sum() / mask_enhanced.size * 100
                _silent_print(f"  🤖 {cam} robot mask: raw {mask_raw_rgb.sum()} px "
                      f"({coverage_raw:.1f}%) → enhanced {mask_enhanced.sum()} px "
                      f"({coverage_enh:.1f}%)")

        # === Step 2: 构建场景点云 (T=0, 三视角拼接, 用增强 mask) ===
        _silent_print(f"\n📐 构建场景点云 (T=0, 三视角拼接, enhanced mask)")
        scene_parts = []
        for cam_idx, cam in enumerate(self.camera_names):
            depth = current_depth[cam_idx]

            pts_cam = self._depth_to_xyz(
                depth, depth_intrinsics[cam], mask_exclude=enhanced_masks_t0[cam]
            )

            if cam != 'front':
                pts_world = self._transform_xyz_to_world(pts_cam, fk_ext_t0[cam])
            else:
                pts_world = pts_cam  # front = world

            scene_parts.append(pts_world)
            _silent_print(f"  {cam}: {len(pts_world)} 场景点")

        scene_xyz = np.concatenate(scene_parts, axis=0)  # [N_scene, 3]
        _silent_print(f"  合计: {len(scene_xyz)} 场景点")

        # === Step 3: 场景点最近邻结构 (GPU 张量 或 KDTree) ===
        if use_gpu:
            scene_xyz_gpu = torch.from_numpy(scene_xyz).float().to(self.device)  # [N_scene, 3]
            scene_kdtree = None
            _silent_print(f"  距离计算: GPU (torch.cdist)")
        else:
            scene_xyz_gpu = None
            scene_kdtree = KDTree(scene_xyz) if len(scene_xyz) > 0 else None
            _silent_print(f"  距离计算: CPU (scipy KDTree)")

        # cond_video2 用的 T=0 scene mask (resize enhanced mask 到 output_size)
        scene_masks_output = []
        for cam_idx, cam in enumerate(self.camera_names):
            enh_mask = enhanced_masks_t0[cam]
            if enh_mask.shape[:2] != output_size:
                rm = cv2.resize(
                    enh_mask.astype(np.uint8), (W_out, H_out),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            else:
                rm = enh_mask
            scene_masks_output.append(~rm)

        # === 预计算 rendermask resize (避免循环内重复 resize) ===
        resized_rendermasks = []  # [K+1][3] → bool mask at output_size
        for t in range(K + 1):
            frame_masks = []
            for cam_idx in range(len(self.camera_names)):
                mask = current_future_rendermask_robosim[t][cam_idx]
                if mask.shape[:2] != output_size:
                    mask = cv2.resize(
                        mask.astype(np.uint8), (W_out, H_out),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                else:
                    mask = mask.astype(bool)
                frame_masks.append(mask)
            resized_rendermasks.append(frame_masks)

        # === Step 4: 逐帧计算 distance maps (两趟: 先算距离, 再统一着色) ===

        # cond_video2: 场景投影只需做一次 (T=0 FK 外参 + 场景点不变)
        scene_proj_cache = []  # 3 views: (valid, u_px, v_px)
        for cam_idx, cam in enumerate(self.camera_names):
            uv, valid = self._project_points_to_2d(
                scene_xyz, fk_ext_t0[cam], output_intrinsics[cam], output_size
            )
            if valid.sum() > 0:
                u_px = np.round(uv[valid, 0]).astype(int)
                v_px = np.round(uv[valid, 1]).astype(int)
            else:
                u_px = np.array([], dtype=int)
                v_px = np.array([], dtype=int)
            scene_proj_cache.append((valid, u_px, v_px))

        _silent_print(f"\n🎯 生成双流 cond video (K={K})")

        # --- Pass 1: 计算所有帧的 dense_map + mask，收集全局距离范围 ---
        dense_maps_1 = []  # K x 3 views: (dense_map, mask)
        dense_maps_2 = []  # K x 3 views: (dense_map, mask)
        all_dists_1 = []  # 收集 cond1 所有有效距离值
        all_dists_2 = []  # 收集 cond2 所有有效距离值

        for k in range(K):
            t = k + 1

            # --- cond_video1: robot → scene 距离 ---
            robot_pts = current_future_points_robosim[t]
            if use_gpu:
                robot_pts_gpu = torch.from_numpy(robot_pts).float().to(self.device)
                dist_r2s = self._gpu_min_dist(
                    robot_pts_gpu, scene_xyz_gpu, chunk_size=self.gpu_dist_chunk_size
                ).cpu().numpy()
                del robot_pts_gpu
            else:
                if len(robot_pts) == 0:
                    dist_r2s = np.empty((0,), dtype=np.float32)
                elif scene_kdtree is None:
                    dist_r2s = np.full((len(robot_pts),), 3.0, dtype=np.float32)
                else:
                    dist_r2s, _ = scene_kdtree.query(robot_pts)

            frame1_data = []
            for cam_idx, cam in enumerate(self.camera_names):
                renderpose = current_future_renderpose_robosim[t][cam_idx]
                uv, valid = self._project_points_to_2d(
                    robot_pts, renderpose, sim_intrinsics[cam], output_size
                )

                sparse_map = np.full((H_out, W_out), np.inf, dtype=np.float64)
                occ = np.zeros((H_out, W_out), dtype=bool)

                if valid.sum() > 0:
                    u_px = np.round(uv[valid, 0]).astype(int)
                    v_px = np.round(uv[valid, 1]).astype(int)
                    dists_valid = dist_r2s[valid]
                    np.minimum.at(sparse_map, (v_px, u_px), dists_valid)
                    occ[v_px, u_px] = True

                rendermask = resized_rendermasks[t][cam_idx]
                sparse_map[~occ] = 0.0
                dense_map = self._sparse_to_dense_in_mask(sparse_map, occ, rendermask)
                frame1_data.append((dense_map, rendermask))

                vals = dense_map[rendermask]
                if len(vals) > 0:
                    all_dists_1.append(vals[np.isfinite(vals)])

            dense_maps_1.append(frame1_data)

            # --- cond_video2: scene → gripper 距离 ---
            gripper_mask = current_future_pointmask_robosim[t]
            gripper_pts = current_future_points_robosim[t][gripper_mask]
            if use_gpu:
                gripper_pts_gpu = torch.from_numpy(gripper_pts).float().to(self.device)
                dist_s2r = self._gpu_min_dist(
                    scene_xyz_gpu, gripper_pts_gpu, chunk_size=self.gpu_dist_chunk_size
                ).cpu().numpy()
                del gripper_pts_gpu
            else:
                if len(scene_xyz) == 0:
                    dist_s2r = np.empty((0,), dtype=np.float32)
                elif len(gripper_pts) == 0:
                    dist_s2r = np.full((len(scene_xyz),), 3.0, dtype=np.float32)
                else:
                    robot_kdtree_k = KDTree(gripper_pts)
                    dist_s2r, _ = robot_kdtree_k.query(scene_xyz)

            frame2_data = []
            for cam_idx, cam in enumerate(self.camera_names):
                valid, u_px, v_px = scene_proj_cache[cam_idx]

                sparse_map = np.full((H_out, W_out), np.inf, dtype=np.float64)
                occ = np.zeros((H_out, W_out), dtype=bool)

                if len(u_px) > 0:
                    dists_valid = dist_s2r[valid]
                    np.minimum.at(sparse_map, (v_px, u_px), dists_valid)
                    occ[v_px, u_px] = True

                scene_mask = scene_masks_output[cam_idx]
                sparse_map[~occ] = 0.0
                dense_map = self._sparse_to_dense_in_mask(sparse_map, occ, scene_mask)
                frame2_data.append((dense_map, scene_mask))

                vals = dense_map[scene_mask]
                if len(vals) > 0:
                    all_dists_2.append(vals[np.isfinite(vals)])

            dense_maps_2.append(frame2_data)

            if (k + 1) % max(1, K // 5) == 0 or k == K - 1:
                r2s_mean = float(np.mean(dist_r2s)) if len(dist_r2s) > 0 else float("nan")
                r2s_max = float(np.max(dist_r2s)) if len(dist_r2s) > 0 else float("nan")
                s2r_mean = float(np.mean(dist_s2r)) if len(dist_s2r) > 0 else float("nan")
                s2r_max = float(np.max(dist_s2r)) if len(dist_s2r) > 0 else float("nan")
                _silent_print(f"  帧 T+{t}: cond1 robot→scene dist "
                      f"mean={r2s_mean:.4f} max={r2s_max:.4f}, "
                      f"cond2 scene→robot dist "
                      f"mean={s2r_mean:.4f} max={s2r_max:.4f}")

        # --- 计算全局归一化 cap (99th percentile across all K frames) ---
        if len(all_dists_1) > 0:
            global_cap_1 = float(np.percentile(np.concatenate(all_dists_1), 99))
        else:
            global_cap_1 = 1.0
        if len(all_dists_2) > 0:
            global_cap_2 = float(np.percentile(np.concatenate(all_dists_2), 99))
        else:
            global_cap_2 = 1.0
        _silent_print(f"  📊 全局归一化 cap: cond1={global_cap_1:.4f}, cond2={global_cap_2:.4f}")

        # --- Pass 2: 用全局 cap 统一着色 ---
        cond_video1 = []
        cond_video2 = []

        for k in range(K):
            frame1_images = []
            for dense_map, mask in dense_maps_1[k]:
                heatmap = self._dist_to_heatmap(dense_map, mask, cap=global_cap_1)
                frame1_images.append(heatmap)
            cond_video1.append(frame1_images)

            frame2_images = []
            for dense_map, mask in dense_maps_2[k]:
                heatmap = self._dist_to_heatmap(dense_map, mask, cap=global_cap_2)
                frame2_images.append(heatmap)
            cond_video2.append(frame2_images)

        _silent_print(f"\n✅ 双流 cond video 生成完成: {K} 帧 x 3 视角")
        if use_gpu:
            del scene_xyz_gpu
        return cond_video1, cond_video2, arm_mask_debug


def _save_cond_results(cond_video1, cond_video2, arm_mask_debug, output_dir):
    """保存 two-stream cond video 结果到文件"""
    camera_names = ['front', 'left_wrist', 'right_wrist']

    # 保存 cond_video1 (robot-centric)
    _silent_print(f"\n💾 保存 cond_video1 (robot-centric) 到 {output_dir}/")
    cond1_dir = output_dir / 'cond_video1'
    cond1_dir.mkdir(exist_ok=True)
    for frame_idx, frame_images in enumerate(cond_video1):
        for cam_idx, cam_name in enumerate(camera_names):
            img = frame_images[cam_idx]
            if img.dtype != np.uint8:
                img = np.clip(img * 255, 0, 255).astype(np.uint8)
            filename = cond1_dir / f'frame_{frame_idx:03d}_{cam_name}.png'
            cv2.imwrite(str(filename), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            _silent_print(f"  ✓ {filename.name}")

    # 保存 cond_video2 (scene-centric)
    _silent_print(f"\n💾 保存 cond_video2 (scene-centric) 到 {output_dir}/")
    cond2_dir = output_dir / 'cond_video2'
    cond2_dir.mkdir(exist_ok=True)
    for frame_idx, frame_images in enumerate(cond_video2):
        for cam_idx, cam_name in enumerate(camera_names):
            img = frame_images[cam_idx]
            if img.dtype != np.uint8:
                img = np.clip(img * 255, 0, 255).astype(np.uint8)
            filename = cond2_dir / f'frame_{frame_idx:03d}_{cam_name}.png'
            cv2.imwrite(str(filename), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            _silent_print(f"  ✓ {filename.name}")

    # 保存 arm_mask_debug (如果有)
    if arm_mask_debug is not None:
        _silent_print(f"\n💾 保存 arm_mask_debug 到 {output_dir}/")
        debug_dir = output_dir / 'arm_mask_debug'
        debug_dir.mkdir(exist_ok=True)
        for cam_name, img in arm_mask_debug.items():
            filename = debug_dir / f'{cam_name}_robot_mask.png'
            cv2.imwrite(str(filename), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            _silent_print(f"  ✓ {filename.name}")


# 使用示例
if __name__ == "__main__":
    import pickle
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Two-stream cond video 生成")
    parser.add_argument('--pkl', type=str,
        default='/mnt/data-2/users/wangboyuan/xxw/episode_0_func_test_inputs_k8.pkl',
        help='pickle 数据路径')
    parser.add_argument('--use_fk', action='store_true',
        help='使用 FK+URDF+mesh 替代 pkl 中的 sim 五项数据')
    parser.add_argument('--output_dir', type=str, default='output_twostream')
    parser.add_argument('--rendermask_dilate', type=int, default=10,
        help='rendermask 膨胀迭代次数')
    parser.add_argument('--no_gpu', action='store_true',
        help='最近邻距离改用 scipy KDTree (CPU)，省显存')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    with open(args.pkl, 'rb') as f:
        data = pickle.load(f)

    generator = CondGenerator(rendermask_dilate_iterations=args.rendermask_dilate)
    data_0 = data[0]

    use_gpu = not args.no_gpu
    if args.use_fk:
        # FK 模式: 只用 pkl 中的 depth + action + obs，sim 五项由 FK 生成
        _silent_print(f"\n📂 FK 模式: 跳过 sim 数据，使用 FK+URDF+mesh")
        cond_video1, cond_video2, arm_mask_debug = generator.depth_point2double_cond(
            current_depth=data_0['current_depth'],
            current_future_action=data_0['current_future_action'],
            current_obs=data_0.get('current_obs'),
            use_gpu=use_gpu,
        )
    else:
        # Sim 模式: 使用 pkl 中全部数据
        _silent_print(f"\n📂 Sim 模式: 使用 pkl 中的 sim 数据")
        cond_video1, cond_video2, arm_mask_debug = generator.depth_point2double_cond(
            **data_0, use_gpu=use_gpu
        )

    _save_cond_results(cond_video1, cond_video2, arm_mask_debug, output_dir)

    # === PLY 点云 ===
    _silent_print(f"\n💾 保存点云 PLY 到 {output_dir}/")
    ply_dir = output_dir / 'ply_debug'
    ply_dir.mkdir(exist_ok=True)

    fk_ext_t0 = generator.forward_kinematics(data_0['current_future_action'][0])
    rgb_hw = (480, 640)
    depth_intrinsics_tmp = {}
    for cam_idx, cam in enumerate(generator.camera_names):
        depth_hw = data_0['current_depth'][cam_idx].shape[:2]
        depth_intrinsics_tmp[cam] = generator._scale_intrinsics(
            generator.intrinsics[cam], rgb_hw, depth_hw
        )

    scene_parts_xyz = []
    scene_parts_rgb = []
    cam_colors = {'front': [200, 200, 200], 'left': [255, 100, 100], 'right': [100, 100, 255]}
    for cam_idx, cam in enumerate(generator.camera_names):
        depth = data_0['current_depth'][cam_idx]
        if args.use_fk:
            # FK 模式: 用 FK 生成的 rendermask
            robot_pts_t0, _ = generator._compute_full_robot_points(data_0['current_future_action'][0])
            output_K = generator._scale_intrinsics(generator.intrinsics[cam], rgb_hw, depth.shape[:2])
            raw_mask = generator._render_robot_mask_from_points(
                robot_pts_t0, fk_ext_t0[cam], output_K, depth.shape[:2]
            )
        else:
            raw_mask = data_0['current_future_rendermask_robosim'][0][cam_idx]
        rgb_for_mask = data_0.get('current_obs', [None] * 3)[cam_idx] if 'current_obs' in data_0 else None
        enhanced_mask = generator._enhance_robot_mask(raw_mask, rgb=rgb_for_mask)
        pts_cam = generator._depth_to_xyz(depth, depth_intrinsics_tmp[cam], mask_exclude=enhanced_mask)
        if cam != 'front':
            pts_world = generator._transform_xyz_to_world(pts_cam, fk_ext_t0[cam])
        else:
            pts_world = pts_cam
        scene_parts_xyz.append(pts_world)
        scene_parts_rgb.append(np.tile(cam_colors[cam], (len(pts_world), 1)))

    scene_xyz_all = np.concatenate(scene_parts_xyz, axis=0)
    scene_rgb_all = np.concatenate(scene_parts_rgb, axis=0).astype(np.uint8)
    scene_cloud = trimesh.PointCloud(vertices=scene_xyz_all, colors=scene_rgb_all)
    scene_cloud.export(str(ply_dir / 'scene_depth_t0.ply'))
    _silent_print(f"  ✓ scene_depth_t0.ply ({len(scene_xyz_all)} 点)")

    # Robot 点云 PLY
    if args.use_fk:
        robot_pts_t0, gripper_mask_t0 = generator._compute_full_robot_points(data_0['current_future_action'][0])
        ply_label = 'robot_fk_t0'
    else:
        robot_pts_t0 = data_0['current_future_points_robosim'][0]
        gripper_mask_t0 = data_0['current_future_pointmask_robosim'][0]
        ply_label = 'robot_sim_t0'

    robot_colors = np.full((len(robot_pts_t0), 3), [180, 180, 180], dtype=np.uint8)
    robot_colors[gripper_mask_t0] = [255, 80, 0]
    robot_cloud = trimesh.PointCloud(vertices=robot_pts_t0, colors=robot_colors)
    robot_cloud.export(str(ply_dir / f'{ply_label}.ply'))
    _silent_print(f"  ✓ {ply_label}.ply ({len(robot_pts_t0)} 点, gripper橙={gripper_mask_t0.sum()})")

    merged_xyz = np.concatenate([scene_xyz_all, robot_pts_t0], axis=0)
    merged_rgb = np.concatenate([scene_rgb_all, robot_colors], axis=0)
    merged_cloud = trimesh.PointCloud(vertices=merged_xyz, colors=merged_rgb)
    merged_cloud.export(str(ply_dir / 'scene_robot_merged_t0.ply'))
    _silent_print(f"  ✓ scene_robot_merged_t0.ply ({len(merged_xyz)} 点)")

    _silent_print(f"\n✅ 所有结果已保存到 {output_dir.resolve()}")
