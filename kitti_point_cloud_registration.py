import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import os
import glob
from scipy.spatial.transform import Rotation
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from sklearn.neighbors import NearestNeighbors
import pickle
import json
import argparse
from pathlib import Path

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class ImprovedKITTIPointCloudMapper:
    def __init__(self, base_path, dataset_prefix, pose_file=None):
        self.base_path = Path(base_path)
        self.date = dataset_prefix
        self.drive = f'{dataset_prefix}_sync'
        
        # 데이터 경로 설정
        self.data_path = self.base_path / 'data' / self.drive
        self.calib_path = self.base_path / 'calibration' / self.date
        
        # 파일 목록 가져오기
        self.velodyne_path = self.data_path / 'velodyne_points' / 'data'
        self.oxts_path = self.data_path / 'oxts' / 'data'
        
        self.velodyne_files = sorted(glob.glob(str(self.velodyne_path / '*.bin')))
        self.oxts_files = sorted(glob.glob(str(self.oxts_path / '*.txt')))
        
        # 사전 계산된 pose 로드
        self.pose_file = pose_file
        self.precomputed_poses = None
        if pose_file:
            self.load_precomputed_poses(pose_file)
        
        # GT vs ICP 비교
        self.gt_aligned_points = []
        self.gt_aligned_colors = []
        self.icp_aligned_points = []
        self.icp_aligned_colors = []
        
        self.frame_poses_gt = []
        self.frame_poses_icp = []
        self.initial_pose = None
        
        # 맵핑 통계
        self.total_points = 0
        self.processing_times = []
        
        # ICP 관련 (사전 계산된 pose가 없을 때만 사용)
        self.icp_cumulative_pose = np.eye(4)
        self.prev_features = None
        self.keyframes = []
        self.keyframe_poses = []
        self.keyframe_features = []
    
    def load_precomputed_poses(self, pose_file):
        """사전 계산된 pose 파일 로드"""
        try:
            print(f"\n사전 계산된 pose 파일 로드 중: {pose_file}")
            with open(pose_file, 'rb') as f:
                self.precomputed_poses = pickle.load(f)
            
            # 디버깅: pose 데이터 구조 확인
            print("\nPose 데이터 구조:")
            print(f"  • 키 목록: {list(self.precomputed_poses.keys())}")
            if 'gt_poses' in self.precomputed_poses:
                print(f"  • GT poses 개수: {len(self.precomputed_poses['gt_poses'])}")
                if len(self.precomputed_poses['gt_poses']) > 0:
                    print(f"  • 첫 번째 GT pose 예시:\n{self.precomputed_poses['gt_poses'][0]}")
            if 'icp_poses' in self.precomputed_poses:
                print(f"  • ICP poses 개수: {len(self.precomputed_poses['icp_poses'])}")
                if len(self.precomputed_poses['icp_poses']) > 0:
                    print(f"  • 첫 번째 ICP pose 예시:\n{self.precomputed_poses['icp_poses'][0]}")
            
            print("✅ Pose 파일 로드 성공!")
            print(f"  • 총 프레임: {len(self.precomputed_poses.get('gt_poses', []))}")
            print(f"  • 알고리즘: {self.precomputed_poses.get('metadata', {}).get('algorithm', 'Unknown')}")
            print(f"  • 생성 시간: {self.precomputed_poses.get('metadata', {}).get('timestamp', 'Unknown')}")
            
        except Exception as e:
            print(f"❌ Pose 파일 로드 실패: {e}")
            self.precomputed_poses = None
    
    def get_pose_from_precomputed(self, frame_idx, pose_type='gt'):
        """사전 계산된 pose에서 특정 프레임의 pose를 가져옴"""
        if self.precomputed_poses is None:
            return None
            
        try:
            if pose_type == 'gt':
                return self.precomputed_poses['gt_poses'][frame_idx]
            elif pose_type == 'icp':
                return self.precomputed_poses['fused_poses'][frame_idx]  # 'icp_poses' -> 'fused_poses'
            else:
                print(f"경고: 알 수 없는 pose 타입: {pose_type}")
                return None
        except (KeyError, IndexError) as e:
            print(f"경고: 프레임 {frame_idx}의 pose를 찾을 수 없습니다: {e}")
            return None
    
    def load_velodyne_points(self, file_path):
        """라이다 포인트 클라우드 로드"""
        points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
        return points
    
    def load_oxts_data(self, file_path):
        """GPS/IMU 데이터 로드"""
        with open(file_path, 'r') as f:
            data = f.readline().strip().split()
        
        return {
            'lat': float(data[0]),
            'lon': float(data[1]),
            'alt': float(data[2]),
            'roll': float(data[3]),
            'pitch': float(data[4]),
            'yaw': float(data[5]),
            'vn': float(data[6]),
            've': float(data[7]),
            'vf': float(data[8])
        }
    
    def lat_lon_to_xyz(self, lat, lon, alt, ref_lat, ref_lon, ref_alt):
        """위도/경도를 XYZ 좌표로 변환"""
        x = (lon - ref_lon) * 111320 * np.cos(np.radians(ref_lat))
        y = (lat - ref_lat) * 111320
        z = alt - ref_alt
        return np.array([x, y, z])
    
    def get_pose_from_oxts(self, frame_idx):
        """OXTS 데이터로부터 GT pose 계산"""
        oxts_data = self.load_oxts_data(self.oxts_files[frame_idx])
        
        if self.initial_pose is None:
            self.initial_pose = {
                'lat': oxts_data['lat'],
                'lon': oxts_data['lon'],
                'alt': oxts_data['alt']
            }
        
        # 위치 계산
        position = self.lat_lon_to_xyz(
            oxts_data['lat'], oxts_data['lon'], oxts_data['alt'],
            self.initial_pose['lat'], self.initial_pose['lon'], self.initial_pose['alt']
        )
        
        # 회전 계산
        rotation = Rotation.from_euler('xyz', [
            oxts_data['roll'],
            oxts_data['pitch'],
            oxts_data['yaw']
        ])
        
        # 4x4 변환 행렬 생성
        pose_matrix = np.eye(4)
        pose_matrix[:3, :3] = rotation.as_matrix()
        pose_matrix[:3, 3] = position
        
        return pose_matrix
    
    def extract_features(self, pcd):
        """포인트 클라우드에서 특징점 추출"""
        # FPFH 특징 계산
        radius_normal = 2.0
        radius_feature = 5.0
        
        # 법선 벡터 계산
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        
        # FPFH 특징 계산
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
        )
        
        return fpfh
    
    def preprocess_point_cloud(self, points, voxel_size=0.1):
        """고급 포인트 클라우드 전처리"""
        # Open3D 포인트 클라우드 생성
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        
        # 1. 지면 제거 (RANSAC 사용)
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.3, ransac_n=3, num_iterations=1000)
        pcd = pcd.select_by_index(inliers, invert=True)
        
        # 2. 통계적 아웃라이어 제거 (더 엄격하게)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        # 3. 다운샘플링 (더 작은 복셀)
        pcd = pcd.voxel_down_sample(voxel_size)
        
        # 4. 반지름 기반 아웃라이어 제거
        pcd, _ = pcd.remove_radius_outlier(nb_points=16, radius=0.5)
        
        return pcd
    
    def multi_scale_icp(self, source, target, scales=[0.5, 0.2, 0.1]):
        """다중 스케일 ICP 정합"""
        current_transformation = np.eye(4)
        
        for scale in scales:
            # 스케일에 따른 다운샘플링
            voxel_size = scale
            source_down = source.voxel_down_sample(voxel_size)
            target_down = target.voxel_down_sample(voxel_size)
            
            # 거리 임계값도 스케일에 맞게 조정
            max_correspondence_distance = scale * 2.0
            
            # Point-to-Plane ICP 수행
            result = o3d.pipelines.registration.registration_icp(
                source_down, target_down,
                max_correspondence_distance=max_correspondence_distance,
                init=current_transformation,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
            )
            
            current_transformation = result.transformation
            
            # 수렴 품질 확인
            if result.fitness < 0.3:  # 품질이 너무 낮으면 이전 변환 유지
                print(f"ICP 품질이 낮음 (fitness: {result.fitness:.3f})")
                break
        
        return current_transformation
    
    def feature_based_initial_alignment(self, source, target, source_fpfh, target_fpfh):
        """특징 기반 초기 정렬"""
        # RANSAC을 사용한 특징 매칭
        distance_threshold = 1.5
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source, target, source_fpfh, target_fpfh,
            mutual_filter=True,
            max_correspondence_distance=distance_threshold,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=3,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
        )
        
        return result.transformation
    
    def detect_loop_closure(self, current_features, frame_idx, threshold=0.85):
        """루프 클로저 검출"""
        if len(self.keyframe_features) < 10:  # 최소 10개 키프레임 필요
            return None, -1
        
        # 현재 프레임과 이전 키프레임들 간의 특징 유사도 계산
        for i, kf_features in enumerate(self.keyframe_features[:-5]):  # 최근 5개는 제외
            if current_features is None or kf_features is None:
                continue
                
            # FPFH 특징 비교
            try:
                current_feat = np.asarray(current_features.data).T
                kf_feat = np.asarray(kf_features.data).T
                
                # 최근접 이웃 매칭
                nbrs = NearestNeighbors(n_neighbors=1).fit(kf_feat)
                distances, indices = nbrs.kneighbors(current_feat)
                
                # 매칭 비율 계산
                good_matches = np.sum(distances < 0.1) / len(distances)
                
                if good_matches > threshold:
                    return self.keyframe_poses[i], i
            except:
                continue
        
        return None, -1
    
    def improved_icp_pose_estimation(self, curr_points, prev_points, prev_features=None):
        """개선된 ICP 기반 pose 추정"""
        if prev_points is None:
            return np.eye(4), None
        
        # 1. 포인트 클라우드 전처리
        source_pcd = self.preprocess_point_cloud(prev_points, voxel_size=0.1)
        target_pcd = self.preprocess_point_cloud(curr_points, voxel_size=0.1)
        
        if len(source_pcd.points) < 100 or len(target_pcd.points) < 100:
            return np.eye(4), None
        
        # 2. 특징 추출
        try:
            source_fpfh = self.extract_features(source_pcd)
            target_fpfh = self.extract_features(target_pcd)
            
            # 3. 특징 기반 초기 정렬
            initial_transform = self.feature_based_initial_alignment(
                source_pcd, target_pcd, source_fpfh, target_fpfh
            )
        except:
            initial_transform = np.eye(4)
            target_fpfh = None
        
        # 4. 다중 스케일 ICP로 정밀 정합
        try:
            final_transform = self.multi_scale_icp(source_pcd, target_pcd, scales=[0.5, 0.2, 0.1])
            
            # 초기 정렬과 정밀 정합 결합
            if not np.allclose(initial_transform, np.eye(4)):
                combined_transform = final_transform @ initial_transform
            else:
                combined_transform = final_transform
                
        except:
            combined_transform = initial_transform
        
        # 5. 변환이 너무 클 경우 필터링 (이상치 제거)
        translation_norm = np.linalg.norm(combined_transform[:3, 3])
        rotation_angle = np.arccos(np.clip((np.trace(combined_transform[:3, :3]) - 1) / 2, -1, 1))
        
        # 합리적인 범위 내의 변환만 허용 (더 엄격하게)
        if translation_norm > 10.0 or rotation_angle > np.pi/6:  # 10m 이상 또는 30도 이상 회전은 거부
            print(f"비현실적인 변환 거부: translation={translation_norm:.2f}m, rotation={np.degrees(rotation_angle):.2f}°")
            combined_transform = np.eye(4)
        
        return combined_transform, target_fpfh
    
    def filter_point_cloud(self, points, intensity_threshold=0.1, distance_range=(3.0, 80.0)):
        """포인트 클라우드 필터링"""
        # 강도 필터링
        intensity_mask = points[:, 3] > intensity_threshold
        
        # 거리 필터링
        distances = np.linalg.norm(points[:, :3], axis=1)
        distance_mask = (distances >= distance_range[0]) & (distances <= distance_range[1])
        
        # 높이 필터링 (지면에서 너무 높거나 낮은 점 제거)
        height_mask = (points[:, 2] > -2.0) & (points[:, 2] < 8.0)
        
        # 전체 마스크 적용
        valid_mask = intensity_mask & distance_mask & height_mask
        filtered_points = points[valid_mask]
        
        return filtered_points
    
    def generate_point_colors(self, points, frame_idx, total_frames, color_scheme='time'):
        """프레임별로 다른 색상 생성"""
        if color_scheme == 'time':
            # 시간에 따른 색상 변화 (파란색 -> 녹색 -> 빨간색)
            color_ratio = frame_idx / max(total_frames - 1, 1)
            hue = (1.0 - color_ratio) * 0.7  # 파란색(0.7)에서 빨간색(0.0)으로
            saturation = 0.8
            value = 0.9
            rgb_color = mcolors.hsv_to_rgb([hue, saturation, value])
            # RGB 값이 0-1 범위를 벗어나지 않도록 클리핑
            rgb_color = np.clip(rgb_color, 0, 1)
        else:
            rgb_color = [0.7, 0.7, 0.7]  # 기본 회색
        
        # 모든 점에 같은 색상 적용
        colors = np.tile(rgb_color, (len(points), 1))
        return colors
    
    def transform_points_to_world(self, points, pose_matrix):
        """포인트를 월드 좌표계로 변환"""
        # 포인트를 homogeneous 좌표로 변환
        points_homo = np.ones((len(points), 4))
        points_homo[:, :3] = points[:, :3]
        
        # 변환 적용
        points_world = (pose_matrix @ points_homo.T).T
        
        return points_world[:, :3]
    
    def create_comparison_maps(self, start_idx=0, end_idx=None, step=2):
        """사전 계산된 pose를 사용한 GT vs ICP 비교"""
        if end_idx is None:
            end_idx = min(len(self.velodyne_files), 100)
        
        # 사전 계산된 pose가 있는지 확인
        if self.precomputed_poses:
            max_frames = self.precomputed_poses['metadata']['total_frames']
            end_idx = min(end_idx, max_frames)
            print(f"사전 계산된 pose 사용 - 최대 프레임: {max_frames}")
        
        print(f"포인트 클라우드 맵 생성 시작...")
        print(f"처리할 프레임: {start_idx}부터 {end_idx}까지 (step: {step})")
        
        prev_points = None
        prev_features = None
        
        for i, frame_idx in enumerate(range(start_idx, end_idx, step)):
            start_time = time.time()
            
            # 포인트 클라우드 로드
            points = self.load_velodyne_points(self.velodyne_files[frame_idx])
            filtered_points = self.filter_point_cloud(points)
            
            # 1. GT pose 기반 정합 (사전 계산된 pose 사용)
            if self.precomputed_poses:
                gt_pose = self.get_pose_from_precomputed(frame_idx, 'gt')
                if gt_pose is None:
                    # fallback to real-time calculation
                    gt_pose = self.get_pose_from_oxts(frame_idx)
            else:
                gt_pose = self.get_pose_from_oxts(frame_idx)
            
            self.frame_poses_gt.append(gt_pose)
            gt_world_points = self.transform_points_to_world(filtered_points, gt_pose)
            gt_colors = self.generate_point_colors(filtered_points, i, (end_idx - start_idx) // step, 'time')
            
            self.gt_aligned_points.append(gt_world_points)
            self.gt_aligned_colors.append(gt_colors)
            
            # 2. 추정된 pose 기반 정합 (사전 계산된 pose 있으면 사용, 없으면 ICP)
            if self.precomputed_poses:
                # 사전 계산된 추정 pose 사용
                estimated_pose = self.get_pose_from_precomputed(frame_idx, 'icp')
                if estimated_pose is None:
                    estimated_pose = np.eye(4)
                self.frame_poses_icp.append(estimated_pose)
                estimated_world_points = self.transform_points_to_world(filtered_points, estimated_pose)
                estimated_colors = self.generate_point_colors(filtered_points, i, (end_idx - start_idx) // step, 'time')
                
                self.icp_aligned_points.append(estimated_world_points)
                self.icp_aligned_colors.append(estimated_colors)
                
            else:
                # 실시간 ICP 수행 (기존 방식)
                if prev_points is not None:
                    icp_relative_pose, curr_features = self.improved_icp_pose_estimation(
                        filtered_points, prev_points, prev_features
                    )
                    
                    # 루프 클로저 검출
                    if curr_features is not None and i % 5 == 0:
                        loop_pose, loop_idx = self.detect_loop_closure(curr_features, frame_idx)
                        if loop_pose is not None:
                            print(f"루프 클로저 검출! 프레임 {frame_idx}에서 키프레임 {loop_idx}와 매칭")
                            self.icp_cumulative_pose = loop_pose
                        
                        self.keyframes.append(frame_idx)
                        self.keyframe_poses.append(self.icp_cumulative_pose.copy())
                        self.keyframe_features.append(curr_features)
                    
                    self.icp_cumulative_pose = self.icp_cumulative_pose @ icp_relative_pose
                    prev_features = curr_features
                else:
                    curr_features = None
                
                self.frame_poses_icp.append(self.icp_cumulative_pose.copy())
                icp_world_points = self.transform_points_to_world(filtered_points, self.icp_cumulative_pose)
                icp_colors = self.generate_point_colors(filtered_points, i, (end_idx - start_idx) // step, 'time')
                
                self.icp_aligned_points.append(icp_world_points)
                self.icp_aligned_colors.append(icp_colors)
            
            # 처리 시간 기록
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            # 진행 상황 출력
            if i % 10 == 0:
                pose_source = "사전계산" if self.precomputed_poses else "실시간"
                print(f"프레임 {frame_idx} 처리 완료 ({pose_source}) - "
                      f"포인트: {len(filtered_points):,}개, "
                      f"시간: {processing_time:.3f}초")
            
            prev_points = filtered_points
            self.total_points += len(filtered_points)
        
        print(f"\n맵 생성 완료! 총 {self.total_points:,}개 포인트")
        if not self.precomputed_poses:
            print(f"키프레임 수: {len(self.keyframes)}")
    
    def create_optimized_point_clouds(self, voxel_size=0.2):
        """최적화된 포인트 클라우드 생성"""
        print("포인트 클라우드 최적화 시작...")
        
        # 사전 계산된 pose 데이터의 범위 확인
        if self.precomputed_poses:
            max_gt_frames = len(self.precomputed_poses.get('gt_poses', []))
            max_icp_frames = len(self.precomputed_poses.get('fused_poses', []))
            max_available_frames = min(max_gt_frames, max_icp_frames, len(self.velodyne_files))
            print(f"  • Velodyne 파일 수: {len(self.velodyne_files)}")
            print(f"  • GT pose 수: {max_gt_frames}")
            print(f"  • ICP pose 수: {max_icp_frames}")
            print(f"  • 처리 가능한 최대 프레임: {max_available_frames}")
        else:
            max_available_frames = len(self.velodyne_files)
        
        # GT 포인트 클라우드 생성
        gt_points = []
        for frame_idx in range(max_available_frames):
            points = self.load_velodyne_points(self.velodyne_files[frame_idx])
            pose = self.get_pose_from_precomputed(frame_idx, pose_type='gt')
            if pose is not None:
                transformed_points = self.transform_points_to_world(points, pose)
                gt_points.append(transformed_points)
            else:
                print(f"⚠️ 프레임 {frame_idx}의 GT pose 데이터가 없습니다.")
        
        if not gt_points:
            print("❌ GT 포인트 클라우드를 생성할 수 없습니다. pose 데이터가 없습니다.")
            return None, None
            
        gt_points = np.vstack(gt_points)
        
        # 추정된 포인트 클라우드 생성
        estimated_points = []
        for frame_idx in range(max_available_frames):
            points = self.load_velodyne_points(self.velodyne_files[frame_idx])
            pose = self.get_pose_from_precomputed(frame_idx, pose_type='icp')
            if pose is not None:
                transformed_points = self.transform_points_to_world(points, pose)
                estimated_points.append(transformed_points)
            else:
                print(f"⚠️ 프레임 {frame_idx}의 추정 pose 데이터가 없습니다.")
        
        if not estimated_points:
            print("❌ 추정된 포인트 클라우드를 생성할 수 없습니다. pose 데이터가 없습니다.")
            return None, None
            
        estimated_points = np.vstack(estimated_points)
        
        # 다운샘플링으로 메모리 및 성능 최적화
        print(f"다운샘플링 시작 (복셀 크기: {voxel_size}m)...")
        
        # GT 포인트 클라우드 다운샘플링
        gt_pcd_temp = o3d.geometry.PointCloud()
        gt_pcd_temp.points = o3d.utility.Vector3dVector(gt_points[:, :3])
        gt_pcd_temp = gt_pcd_temp.voxel_down_sample(voxel_size)
        gt_points_downsampled = np.asarray(gt_pcd_temp.points)
        
        # 추정된 포인트 클라우드 다운샘플링
        est_pcd_temp = o3d.geometry.PointCloud()
        est_pcd_temp.points = o3d.utility.Vector3dVector(estimated_points[:, :3])
        est_pcd_temp = est_pcd_temp.voxel_down_sample(voxel_size)
        estimated_points_downsampled = np.asarray(est_pcd_temp.points)
        
        print(f"다운샘플링 완료:")
        print(f"  • GT 포인트: {len(gt_points):,} → {len(gt_points_downsampled):,}")
        print(f"  • 추정 포인트: {len(estimated_points):,} → {len(estimated_points_downsampled):,}")
        
        # Open3D 포인트 클라우드 생성 (다운샘플링된 버전)
        self.gt_pcd = o3d.geometry.PointCloud()
        self.gt_pcd.points = o3d.utility.Vector3dVector(gt_points_downsampled)
        
        # 시간에 따른 그라데이션 색상 생성
        num_points = len(gt_points_downsampled)
        colors = np.zeros((num_points, 3))
        for i in range(num_points):
            # 시간에 따른 색상 변화 (파란색 -> 녹색 -> 빨간색)
            ratio = i / num_points
            if ratio < 0.5:
                # 파란색 -> 녹색
                colors[i] = [0, ratio * 2, 1 - ratio * 2]
            else:
                # 녹색 -> 빨간색
                colors[i] = [(ratio - 0.5) * 2, 1 - (ratio - 0.5) * 2, 0]
        
        self.gt_pcd.colors = o3d.utility.Vector3dVector(colors)
        
        self.icp_pcd = o3d.geometry.PointCloud()
        self.icp_pcd.points = o3d.utility.Vector3dVector(estimated_points_downsampled)
        
        # ICP 포인트 클라우드도 동일한 그라데이션 적용
        num_points = len(estimated_points_downsampled)
        colors = np.zeros((num_points, 3))
        for i in range(num_points):
            ratio = i / num_points
            if ratio < 0.5:
                colors[i] = [0, ratio * 2, 1 - ratio * 2]
            else:
                colors[i] = [(ratio - 0.5) * 2, 1 - (ratio - 0.5) * 2, 0]
        
        self.icp_pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # 결과 저장
        results_dir = Path('results') / self.date
        os.makedirs(results_dir, exist_ok=True)
        
        # GT 포인트 클라우드 저장
        gt_filepath = results_dir / 'kitti_gt_pointcloud.pcd'
        o3d.io.write_point_cloud(str(gt_filepath), self.gt_pcd)
        print(f"✅ GT 포인트 클라우드가 저장되었습니다: {gt_filepath}")
        
        # ICP 포인트 클라우드 저장
        icp_filepath = results_dir / 'kitti_estimated_pointcloud.pcd'
        o3d.io.write_point_cloud(str(icp_filepath), self.icp_pcd)
        print(f"✅ 추정된 포인트 클라우드가 저장되었습니다: {icp_filepath}")
        
        # 궤적 데이터 저장
        trajectory_data = {
            'gt_trajectory': np.array([self.get_pose_from_precomputed(i, pose_type='gt')[:3, 3] 
                                     for i in range(max_available_frames)
                                     if self.get_pose_from_precomputed(i, pose_type='gt') is not None]),
            'icp_trajectory': np.array([self.get_pose_from_precomputed(i, pose_type='icp')[:3, 3] 
                                      for i in range(max_available_frames)
                                      if self.get_pose_from_precomputed(i, pose_type='icp') is not None])
        }
        trajectory_filepath = results_dir / 'kitti_trajectory.npz'
        np.savez(trajectory_filepath, **trajectory_data)
        print(f"✅ 궤적 데이터가 저장되었습니다: {trajectory_filepath}")
        
        print("포인트 클라우드 최적화 완료!")
        
        return gt_points_downsampled, estimated_points_downsampled
    
    def visualize_comparison_2d(self):
        """GT vs 추정된 pose 비교 시각화"""
        if not hasattr(self, 'gt_pcd') or not hasattr(self, 'icp_pcd'):
            print("❌ 먼저 create_test_map_simple() 함수를 실행해서 포인트 클라우드를 생성해주세요.")
            return
        
        if not hasattr(self, 'frame_poses_gt') or not hasattr(self, 'frame_poses_icp'):
            print("❌ pose 데이터가 없습니다. 먼저 포인트 클라우드를 생성해주세요.")
            return
        
        print("2D 시각화 시작...")
        
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        
        # 1. GT pose 기반 정합
        gt_points = np.asarray(self.gt_pcd.points)
        gt_colors = np.asarray(self.gt_pcd.colors)
        gt_trajectory = np.array([pose[:3, 3] for pose in self.frame_poses_gt])
        
        # 포인트 수가 너무 많으면 추가로 샘플링
        if len(gt_points) > 50000:
            sample_indices = np.random.choice(len(gt_points), 50000, replace=False)
            gt_points_vis = gt_points[sample_indices]
            gt_colors_vis = gt_colors[sample_indices]
            print(f"GT 포인트 시각화용 샘플링: {len(gt_points):,} → {len(gt_points_vis):,}")
        else:
            gt_points_vis = gt_points
            gt_colors_vis = gt_colors
        
        axes[0].scatter(gt_points_vis[:, 0], gt_points_vis[:, 1], c=gt_colors_vis, s=0.1, alpha=0.7)
        axes[0].plot(gt_trajectory[:, 0], gt_trajectory[:, 1], 'r-', linewidth=3, label='GT Trajectory', alpha=0.9)
        axes[0].scatter(gt_trajectory[0, 0], gt_trajectory[0, 1], c='green', s=150, marker='o', label='Start', zorder=5)
        axes[0].scatter(gt_trajectory[-1, 0], gt_trajectory[-1, 1], c='red', s=150, marker='X', label='End', zorder=5)
        axes[0].set_title('GT Pose-based Mapping (Ground Truth)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('X Position (m)')
        axes[0].set_ylabel('Y Position (m)')
        axes[0].grid(True, alpha=0.3)
        axes[0].axis('equal')
        axes[0].legend()
        
        # 2. 추정된 pose 기반 정합
        estimated_points = np.asarray(self.icp_pcd.points)
        estimated_colors = np.asarray(self.icp_pcd.colors)
        estimated_trajectory = np.array([pose[:3, 3] for pose in self.frame_poses_icp])
        
        # 포인트 수가 너무 많으면 추가로 샘플링
        if len(estimated_points) > 50000:
            sample_indices = np.random.choice(len(estimated_points), 50000, replace=False)
            estimated_points_vis = estimated_points[sample_indices]
            estimated_colors_vis = estimated_colors[sample_indices]
            print(f"추정 포인트 시각화용 샘플링: {len(estimated_points):,} → {len(estimated_points_vis):,}")
        else:
            estimated_points_vis = estimated_points
            estimated_colors_vis = estimated_colors
        
        pose_method = "LIO (ICP) + GPS/IMU Fusion" if self.precomputed_poses else "Real-time ICP"
        
        axes[1].scatter(estimated_points_vis[:, 0], estimated_points_vis[:, 1], c=estimated_colors_vis, s=0.1, alpha=0.7)
        axes[1].plot(estimated_trajectory[:, 0], estimated_trajectory[:, 1], 'b-', linewidth=3, 
                    label=f'{pose_method} Trajectory', alpha=0.9)
        axes[1].scatter(estimated_trajectory[0, 0], estimated_trajectory[0, 1], c='green', s=150, marker='o', label='Start', zorder=5)
        axes[1].scatter(estimated_trajectory[-1, 0], estimated_trajectory[-1, 1], c='red', s=150, marker='X', label='End', zorder=5)
        
        # 키프레임 표시 (실시간 ICP인 경우에만)
        if not self.precomputed_poses and hasattr(self, 'keyframes') and self.keyframes:
            keyframe_positions = np.array([pose[:3, 3] for pose in self.keyframe_poses])
            axes[1].scatter(keyframe_positions[:, 0], keyframe_positions[:, 1], 
                           c='orange', s=100, marker='D', label='Keyframes', zorder=4, alpha=0.8)
        
        axes[1].set_title(f'{pose_method} Mapping', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('X Position (m)')
        axes[1].set_ylabel('Y Position (m)')
        axes[1].grid(True, alpha=0.3)
        axes[1].axis('equal')
        axes[1].legend()
        
        # 3. 궤적 비교 + 오차
        axes[2].plot(gt_trajectory[:, 0], gt_trajectory[:, 1], 'r-', linewidth=4, label='GT Trajectory', alpha=0.8)
        axes[2].plot(estimated_trajectory[:, 0], estimated_trajectory[:, 1], 'b-', linewidth=4, 
                    label=f'{pose_method} Trajectory', alpha=0.8)
        
        # 오차 벡터 표시 (일부만)
        step_error = max(1, len(gt_trajectory) // 20)
        for i in range(0, len(gt_trajectory), step_error):
            if i < len(estimated_trajectory):  # 안전성 체크
                axes[2].plot([gt_trajectory[i, 0], estimated_trajectory[i, 0]], 
                            [gt_trajectory[i, 1], estimated_trajectory[i, 1]], 
                            'k-', alpha=0.3, linewidth=1)
        
        axes[2].scatter(gt_trajectory[0, 0], gt_trajectory[0, 1], c='green', s=200, marker='o', label='Start', zorder=5)
        axes[2].scatter(gt_trajectory[-1, 0], gt_trajectory[-1, 1], c='red', s=200, marker='X', label='End', zorder=5)
        
        # 오차 통계 표시 (궤적 길이를 맞춤)
        min_length = min(len(gt_trajectory), len(estimated_trajectory))
        trajectory_errors = np.linalg.norm(gt_trajectory[:min_length] - estimated_trajectory[:min_length], axis=1)
        mean_error = np.mean(trajectory_errors)
        max_error = np.max(trajectory_errors)
        
        axes[2].text(0.02, 0.98, f'Mean Error: {mean_error:.2f}m\nMax Error: {max_error:.2f}m', 
                    transform=axes[2].transAxes, fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        axes[2].set_title('Trajectory Comparison & Error Analysis', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('X Position (m)')
        axes[2].set_ylabel('Y Position (m)')
        axes[2].grid(True, alpha=0.3)
        axes[2].axis('equal')
        axes[2].legend()
        
        plt.tight_layout()
        
        # 결과 저장
        results_dir = Path('results') / self.date
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f'kitti_point_cloud_comparison_2d_{timestamp}.png'
        filepath = results_dir / filename
        
        print("그래프 저장 중...")
        try:
            plt.savefig(filepath, dpi=150, bbox_inches='tight')  # DPI를 낮춰서 성능 개선
            print(f"✅ 2D 비교 결과가 저장되었습니다: {filepath}")
        except Exception as e:
            print(f"⚠️ 그래프 저장 중 오류 발생: {e}")
        
        print("그래프 표시 중...")
        plt.show()
    
    def analyze_improved_performance(self):
        """성능 분석 (사전계산 vs 실시간)"""
        if not hasattr(self, 'gt_pcd'):
            print("먼저 포인트 클라우드를 생성해주세요.")
            return
        
        # 궤적 오차 계산
        gt_trajectory = np.array([pose[:3, 3] for pose in self.frame_poses_gt])
        estimated_trajectory = np.array([pose[:3, 3] for pose in self.frame_poses_icp])
        
        trajectory_errors = np.linalg.norm(gt_trajectory - estimated_trajectory, axis=1)
        
        # 포인트 클라우드 통계
        gt_points = np.asarray(self.gt_pcd.points)
        estimated_points = np.asarray(self.icp_pcd.points)
        
        # 방법 결정
        method_name = "LIO (ICP) + GPS/IMU Fusion" if self.precomputed_poses else "Real-time ICP"
        
        # 리포트 출력
        print("\n" + "="*85)
        print("                    POINT CLOUD MAPPING PERFORMANCE REPORT")
        print("="*85)
        
        print("DATASET INFORMATION:")
        print(f"  • Total Frames Processed:       {len(self.frame_poses_gt)}")
        print(f"  • Total Points:                 {self.total_points:,}")
        if not self.precomputed_poses:
            print(f"  • Keyframes Generated:          {len(self.keyframes)}")
        print(f"  • Processing Method:            GT vs {method_name}")
        print(f"  • Pose Source:                  {'LIO (ICP) + GPS/IMU Fusion' if self.precomputed_poses else 'Real-time ICP'}")
        
        print("-"*85)
        print("POINT CLOUD STATISTICS:")
        print(f"  • GT Aligned Points:            {len(gt_points):,}")
        print(f"  • Estimated Aligned Points:     {len(estimated_points):,}")
        print(f"  • Map Consistency:              {abs(len(gt_points) - len(estimated_points)) / len(gt_points) * 100:.1f}% difference")
        
        print("-"*85)
        print("TRAJECTORY ACCURACY ANALYSIS:")
        print(f"  • Mean Trajectory Error:        {np.mean(trajectory_errors):.3f} m")
        print(f"  • Median Trajectory Error:      {np.median(trajectory_errors):.3f} m")
        print(f"  • Max Trajectory Error:         {np.max(trajectory_errors):.3f} m")
        print(f"  • Min Trajectory Error:         {np.min(trajectory_errors):.3f} m")
        print(f"  • RMSE Trajectory Error:        {np.sqrt(np.mean(trajectory_errors**2)):.3f} m")
        print(f"  • Standard Deviation:           {np.std(trajectory_errors):.3f} m")
        
        # 오차 백분위수
        p25 = np.percentile(trajectory_errors, 25)
        p75 = np.percentile(trajectory_errors, 75)
        p95 = np.percentile(trajectory_errors, 95)
        
        print(f"  • 25th Percentile Error:        {p25:.3f} m")
        print(f"  • 75th Percentile Error:        {p75:.3f} m")
        print(f"  • 95th Percentile Error:        {p95:.3f} m")
        
        # GT vs 추정 총 이동 거리 비교
        gt_total_distance = np.sum(np.linalg.norm(np.diff(gt_trajectory, axis=0), axis=1))
        estimated_total_distance = np.sum(np.linalg.norm(np.diff(estimated_trajectory, axis=0), axis=1))
        
        print("-"*85)
        print("DISTANCE ANALYSIS:")
        print(f"  • GT Total Distance:            {gt_total_distance:.1f} m")
        print(f"  • Estimated Total Distance:     {estimated_total_distance:.1f} m")
        print(f"  • Distance Difference:          {abs(gt_total_distance - estimated_total_distance):.1f} m")
        print(f"  • Distance Accuracy:            {100 - abs(gt_total_distance - estimated_total_distance)/gt_total_distance*100:.1f}%")
        
        print("-"*85)
        if self.precomputed_poses:
            print("PRE-COMPUTED POSE FEATURES:")
            print("  • Algorithm Used:               LIO (ICP) + GPS/IMU Sensor Fusion")
            print("  • Kalman Filter:                ✓ Extended Kalman Filter")
            print("  • Real-time Processing:         ✓ Pre-computed offline")
            print("  • Computational Efficiency:     ✓ High (loading from file)")
            print("  • Consistency:                  ✓ Identical results every run")
        else:
            print("REAL-TIME ICP FEATURES:")
            print("  • Multi-scale ICP:              ✓ Coarse-to-fine registration")
            print("  • Feature-based Initialization: ✓ FPFH features + RANSAC")
            print("  • Statistical Outlier Removal:  ✓ Advanced preprocessing")
            print("  • Loop Closure Detection:       ✓ Feature matching based")
            print("  • Motion Validation:            ✓ Unrealistic motion filtering")
        
        print("-"*85)
        print("PERFORMANCE ASSESSMENT:")
        
        # 품질 평가
        mean_error = np.mean(trajectory_errors)
        if mean_error < 2.0:
            quality_grade = "EXCELLENT"
            description = "Commercial-grade accuracy"
        elif mean_error < 5.0:
            quality_grade = "VERY GOOD"
            description = "Research-grade system"
        elif mean_error < 10.0:
            quality_grade = "GOOD"
            description = "Acceptable for most applications"
        elif mean_error < 20.0:
            quality_grade = "FAIR"
            description = "Needs further optimization"
        else:
            quality_grade = "POOR"
            description = "Significant improvements required"
        
        print(f"  • Overall Quality Grade:        {quality_grade}")
        print(f"  • System Assessment:            {description}")
        print(f"  • Drift Rate:                   {mean_error/gt_total_distance*100:.3f}% per meter")
        print(f"  • Suitable for Autonomous Nav:  {'YES' if mean_error < 5.0 else 'NO'}")
        print(f"  • Suitable for Mapping:         {'YES' if mean_error < 10.0 else 'NO'}")
        
        print("-"*85)
        print("IMPROVEMENT RECOMMENDATIONS:")
        if self.precomputed_poses:
            if mean_error > 2.0:
                print("  • Consider improving the pose estimation algorithm")
                print("  • Add more sophisticated sensor fusion techniques")
            print("  • Advantage: Consistent and reproducible results")
            print("  • Advantage: Fast mapping execution")
        else:
            if mean_error > 5.0:
                print("  • Consider adding IMU integration for better motion prediction")
                print("  • Implement more sophisticated loop closure detection")
                print("  • Add bundle adjustment for global optimization")
            if len(self.keyframes) < 10:
                print("  • Increase keyframe frequency for better loop closure")
            if np.std(trajectory_errors) > mean_error:
                print("  • High error variance - consider adaptive parameters")
        
        print("="*85)
        print(f"Point Cloud Mapping with {method_name} completed successfully!")
        print("="*85)
        
        # results 폴더 생성
        results_dir = 'results'
        os.makedirs(results_dir, exist_ok=True)
        
        # 성능 리포트를 텍스트 파일로 저장
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        method_name = "lio_gps_imu" if self.precomputed_poses else "realtime_icp"
        report_filename = f'kitti_mapping_report_{method_name}_{timestamp}.txt'
        report_filepath = os.path.join(results_dir, report_filename)
        
        # 현재 출력을 파일로 리다이렉트
        import sys
        original_stdout = sys.stdout
        with open(report_filepath, 'w', encoding='utf-8') as f:
            sys.stdout = f
            # 리포트 내용을 직접 파일에 작성
            f.write("\n" + "="*85 + "\n")
            f.write("                    POINT CLOUD MAPPING PERFORMANCE REPORT\n")
            f.write("="*85 + "\n\n")
            
            f.write("DATASET INFORMATION:\n")
            f.write(f"  • Total Frames Processed:       {len(self.frame_poses_gt)}\n")
            f.write(f"  • Total Points:                 {self.total_points:,}\n")
            if not self.precomputed_poses:
                f.write(f"  • Keyframes Generated:          {len(self.keyframes)}\n")
            f.write(f"  • Processing Method:            GT vs {method_name}\n")
            f.write(f"  • Pose Source:                  {'LIO (ICP) + GPS/IMU Fusion' if self.precomputed_poses else 'Real-time ICP'}\n")
            
            f.write("-"*85 + "\n")
            f.write("POINT CLOUD STATISTICS:\n")
            f.write(f"  • GT Aligned Points:            {len(gt_points):,}\n")
            f.write(f"  • Estimated Aligned Points:     {len(estimated_points):,}\n")
            f.write(f"  • Map Consistency:              {abs(len(gt_points) - len(estimated_points)) / len(gt_points) * 100:.1f}% difference\n")
            
            f.write("-"*85 + "\n")
            f.write("TRAJECTORY ACCURACY ANALYSIS:\n")
            f.write(f"  • Mean Trajectory Error:        {np.mean(trajectory_errors):.3f} m\n")
            f.write(f"  • Median Trajectory Error:      {np.median(trajectory_errors):.3f} m\n")
            f.write(f"  • Max Trajectory Error:         {np.max(trajectory_errors):.3f} m\n")
            f.write(f"  • Min Trajectory Error:         {np.min(trajectory_errors):.3f} m\n")
            f.write(f"  • RMSE Trajectory Error:        {np.sqrt(np.mean(trajectory_errors**2)):.3f} m\n")
            f.write(f"  • Standard Deviation:           {np.std(trajectory_errors):.3f} m\n")
            
            # 오차 백분위수
            f.write("-"*85 + "\n")
            f.write("TRAJECTORY ACCURACY ANALYSIS:\n")
            f.write(f"  • 25th Percentile Error:        {p25:.3f} m\n")
            f.write(f"  • 75th Percentile Error:        {p75:.3f} m\n")
            f.write(f"  • 95th Percentile Error:        {p95:.3f} m\n")
            
            f.write("-"*85 + "\n")
            f.write("DISTANCE ANALYSIS:\n")
            f.write(f"  • GT Total Distance:            {gt_total_distance:.1f} m\n")
            f.write(f"  • Estimated Total Distance:     {estimated_total_distance:.1f} m\n")
            f.write(f"  • Distance Difference:          {abs(gt_total_distance - estimated_total_distance):.1f} m\n")
            f.write(f"  • Distance Accuracy:            {100 - abs(gt_total_distance - estimated_total_distance)/gt_total_distance*100:.1f}%\n")
            
            f.write("-"*85 + "\n")
            if self.precomputed_poses:
                f.write("PRE-COMPUTED POSE FEATURES:\n")
                f.write("  • Algorithm Used:               LIO (ICP) + GPS/IMU Sensor Fusion\n")
                f.write("  • Kalman Filter:                ✓ Extended Kalman Filter\n")
                f.write("  • Real-time Processing:         ✓ Pre-computed offline\n")
                f.write("  • Computational Efficiency:     ✓ High (loading from file)\n")
                f.write("  • Consistency:                  ✓ Identical results every run\n")
            else:
                f.write("REAL-TIME ICP FEATURES:\n")
                f.write("  • Multi-scale ICP:              ✓ Coarse-to-fine registration\n")
                f.write("  • Feature-based Initialization: ✓ FPFH features + RANSAC\n")
                f.write("  • Statistical Outlier Removal:  ✓ Advanced preprocessing\n")
                f.write("  • Loop Closure Detection:       ✓ Feature matching based\n")
                f.write("  • Motion Validation:            ✓ Unrealistic motion filtering\n")
            
            f.write("-"*85 + "\n")
            f.write("PERFORMANCE ASSESSMENT:\n")
            
            # 품질 평가
            f.write(f"  • Overall Quality Grade:        {quality_grade}\n")
            f.write(f"  • System Assessment:            {description}\n")
            f.write(f"  • Drift Rate:                   {mean_error/gt_total_distance*100:.3f}% per meter\n")
            f.write(f"  • Suitable for Autonomous Nav:  {'YES' if mean_error < 5.0 else 'NO'}\n")
            f.write(f"  • Suitable for Mapping:         {'YES' if mean_error < 10.0 else 'NO'}\n")
            
            f.write("-"*85 + "\n")
            f.write("IMPROVEMENT RECOMMENDATIONS:\n")
            if self.precomputed_poses:
                if mean_error > 2.0:
                    f.write("  • Consider improving the pose estimation algorithm\n")
                    f.write("  • Add more sophisticated sensor fusion techniques\n")
                f.write("  • Advantage: Consistent and reproducible results\n")
                f.write("  • Advantage: Fast mapping execution\n")
            else:
                if mean_error > 5.0:
                    f.write("  • Consider adding IMU integration for better motion prediction\n")
                    f.write("  • Implement more sophisticated loop closure detection\n")
                    f.write("  • Add bundle adjustment for global optimization\n")
                if len(self.keyframes) < 10:
                    f.write("  • Increase keyframe frequency for better loop closure\n")
                if np.std(trajectory_errors) > mean_error:
                    f.write("  • High error variance - consider adaptive parameters\n")
            
            f.write("="*85 + "\n")
            f.write(f"Point Cloud Mapping with {method_name} completed successfully!\n")
            f.write("="*85 + "\n")
            
            sys.stdout = original_stdout
        
        print(f"\n성능 리포트가 저장되었습니다: {report_filepath}")

    def debug_pose_and_transformation(self, frame_idx, points, pose_matrix, method_name):
        """pose와 변환 결과 디버깅"""
        print(f"\n=== {method_name} - 프레임 {frame_idx} 디버깅 ===")
        print(f"Pose 행렬:")
        print(pose_matrix)
        print(f"위치: {pose_matrix[:3, 3]}")
        
        # 몇 개 포인트만 변환해서 확인
        sample_points = points[:5]  # 첫 5개 포인트
        transformed_points = self.transform_points_to_world(sample_points, pose_matrix)
        
        print(f"원본 포인트 (첫 5개):")
        for i, p in enumerate(sample_points):
            print(f"  {i}: [{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}]")
        
        print(f"변환된 포인트 (첫 5개):")
        for i, p in enumerate(transformed_points):
            print(f"  {i}: [{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}]")
        
        print(f"변환량: {np.linalg.norm(pose_matrix[:3, 3]):.3f}m")
        print("=" * 50)
    
    def create_test_map_simple(self, max_frames=None):
        """간단한 테스트 맵 생성"""
        print("🔍 포인트클라우드 정합 문제 진단 시작...")
        
        # 프레임 범위 설정
        if max_frames is None:
            max_frames = len(self.velodyne_files)
        
        # 사전 계산된 pose 데이터의 범위 확인
        if self.precomputed_poses:
            max_gt_frames = len(self.precomputed_poses.get('gt_poses', []))
            max_icp_frames = len(self.precomputed_poses.get('fused_poses', []))
            max_available_frames = min(max_gt_frames, max_icp_frames, max_frames, len(self.velodyne_files))
            print(f"  • 처리 가능한 최대 프레임: {max_available_frames}")
        
        # 프레임별 포인트 클라우드 저장
        frame_points = []
        frame_poses_gt = []
        frame_poses_icp = []
        
        # 2프레임 간격으로 처리
        for frame_idx in range(0, max_frames, 2):
            # 현재 프레임의 라이다 포인트 로드
            points = self.load_velodyne_points(self.velodyne_files[frame_idx])
            
            # GT pose와 ICP pose 가져오기
            gt_pose = self.get_pose_from_precomputed(frame_idx, pose_type='gt')
            icp_pose = self.get_pose_from_precomputed(frame_idx, pose_type='icp')
            
            if gt_pose is not None and icp_pose is not None:
                # 포인트 클라우드를 월드 좌표계로 변환
                gt_points = self.transform_points_to_world(points, gt_pose)
                icp_points = self.transform_points_to_world(points, icp_pose)
                
                frame_points.append((gt_points, icp_points))
                frame_poses_gt.append(gt_pose)
                frame_poses_icp.append(icp_pose)
            else:
                print(f"⚠️ 프레임 {frame_idx}의 pose 데이터가 없습니다.")
        
        if not frame_points:
            print("❌ 처리할 수 있는 프레임이 없습니다.")
            return
        
        # 포인트 클라우드 통합
        gt_points = np.vstack([points[0] for points in frame_points])
        icp_points = np.vstack([points[1] for points in frame_points])
        
        # 다운샘플링으로 성능 최적화
        voxel_size = 0.2
        print(f"다운샘플링 시작 (복셀 크기: {voxel_size}m)...")
        
        # GT 포인트 클라우드 다운샘플링
        gt_pcd_temp = o3d.geometry.PointCloud()
        gt_pcd_temp.points = o3d.utility.Vector3dVector(gt_points[:, :3])
        gt_pcd_temp = gt_pcd_temp.voxel_down_sample(voxel_size)
        gt_points_downsampled = np.asarray(gt_pcd_temp.points)
        
        # ICP 포인트 클라우드 다운샘플링
        icp_pcd_temp = o3d.geometry.PointCloud()
        icp_pcd_temp.points = o3d.utility.Vector3dVector(icp_points[:, :3])
        icp_pcd_temp = icp_pcd_temp.voxel_down_sample(voxel_size)
        icp_points_downsampled = np.asarray(icp_pcd_temp.points)
        
        print(f"다운샘플링 완료:")
        print(f"  • GT 포인트: {len(gt_points):,} → {len(gt_points_downsampled):,}")
        print(f"  • ICP 포인트: {len(icp_points):,} → {len(icp_points_downsampled):,}")
        
        # Open3D 포인트 클라우드 생성 (다운샘플링된 버전)
        self.gt_pcd = o3d.geometry.PointCloud()
        self.gt_pcd.points = o3d.utility.Vector3dVector(gt_points_downsampled)
        
        # 시간에 따른 그라데이션 색상 생성
        num_points = len(gt_points_downsampled)
        colors = np.zeros((num_points, 3))
        for i in range(num_points):
            # 시간에 따른 색상 변화 (파란색 -> 녹색 -> 빨간색)
            ratio = i / num_points
            if ratio < 0.5:
                # 파란색 -> 녹색
                colors[i] = [0, ratio * 2, 1 - ratio * 2]
            else:
                # 녹색 -> 빨간색
                colors[i] = [(ratio - 0.5) * 2, 1 - (ratio - 0.5) * 2, 0]
        
        self.gt_pcd.colors = o3d.utility.Vector3dVector(colors)
        
        self.icp_pcd = o3d.geometry.PointCloud()
        self.icp_pcd.points = o3d.utility.Vector3dVector(icp_points_downsampled)
        
        # ICP 포인트 클라우드도 동일한 그라데이션 적용
        num_points = len(icp_points_downsampled)
        colors = np.zeros((num_points, 3))
        for i in range(num_points):
            ratio = i / num_points
            if ratio < 0.5:
                colors[i] = [0, ratio * 2, 1 - ratio * 2]
            else:
                colors[i] = [(ratio - 0.5) * 2, 1 - (ratio - 0.5) * 2, 0]
        
        self.icp_pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # pose 저장
        self.frame_poses_gt = frame_poses_gt
        self.frame_poses_icp = frame_poses_icp
        
        print("✅ 테스트 맵 생성 완료!")
        print(f"  • 처리된 프레임 수: {len(frame_points)}")
        print(f"  • GT 포인트 수: {len(gt_points_downsampled):,}")
        print(f"  • ICP 포인트 수: {len(icp_points_downsampled):,}")
        
        # 포인트 클라우드 범위 출력
        gt_min = np.min(gt_points_downsampled, axis=0)
        gt_max = np.max(gt_points_downsampled, axis=0)
        print("\n포인트 클라우드 범위:")
        print(f"  • X 범위: [{gt_min[0]:.2f}, {gt_max[0]:.2f}]")
        print(f"  • Y 범위: [{gt_min[1]:.2f}, {gt_max[1]:.2f}]")
        print(f"  • Z 범위: [{gt_min[2]:.2f}, {gt_max[2]:.2f}]")

    def visualize_camera_lidar_fusion(self, start_frame=0, num_frames=5, frame_interval=10):
        """카메라-라이다 정합 시각화 (여러 프레임)"""
        print("\n✅ 카메라-라이다 정합 시각화 시작...")
        
        # 1. 캘리브레이션 파일 로드
        calib_path = os.path.join(self.base_path, self.date, 'calib_cam_to_cam.txt')
        try:
            with open(calib_path, 'r') as f:
                calib_data = f.readlines()
            
            # 캘리브레이션 행렬 파싱
            # P2 (카메라 프로젝션 행렬)
            P2_line = [line for line in calib_data if line.startswith('P_rect_02')][0]
            P2 = np.array([float(x) for x in P2_line.strip().split()[1:13]]).reshape(3, 4)
            
            # R0_rect (정규화 행렬)
            R0_line = [line for line in calib_data if line.startswith('R_rect_02')][0]
            R0_rect = np.array([float(x) for x in R0_line.strip().split()[1:10]]).reshape(3, 3)
            
            # 라이다->카메라 변환 행렬 로드
            velo_to_cam_path = os.path.join(self.base_path, self.date, 'calib_velo_to_cam.txt')
            with open(velo_to_cam_path, 'r') as f:
                velo_to_cam_data = f.readlines()
            
            # R (회전 행렬)
            R_line = [line for line in velo_to_cam_data if line.startswith('R:')][0]
            R = np.array([float(x) for x in R_line.strip().split()[1:10]]).reshape(3, 3)
            
            # T (이동 벡터)
            T_line = [line for line in velo_to_cam_data if line.startswith('T:')][0]
            T = np.array([float(x) for x in T_line.strip().split()[1:4]]).reshape(3, 1)
            
            # Tr_velo_to_cam 생성 (3x4 변환 행렬)
            Tr_velo_to_cam = np.hstack((R, T))
            
            print("✅ 캘리브레이션 파일 로드 성공")
        except Exception as e:
            print(f"❌ 캘리브레이션 파일 로드 실패: {e}")
            return
        
        # 2. 여러 프레임 시각화
        for i in range(num_frames):
            frame_idx = start_frame + i * frame_interval
            
            # 이미지 로드
            image_path = os.path.join(self.base_path, self.drive, 'image_02', 'data', f'{frame_idx:010d}.png')
            try:
                image = plt.imread(image_path)
                print(f"✅ 이미지 로드 성공: {image_path}")
            except Exception as e:
                print(f"❌ 이미지 로드 실패: {e}")
                continue
            
            # 라이다 포인트 로드 및 필터링
            points = self.load_velodyne_points(self.velodyne_files[frame_idx])
            filtered_points = self.filter_point_cloud(points)
            
            # 포인트 클라우드를 카메라 좌표계로 변환
            points_homo = np.ones((len(filtered_points), 4))
            points_homo[:, :3] = filtered_points[:, :3]
            
            # 라이다 -> 카메라 변환
            points_cam = (Tr_velo_to_cam @ points_homo.T).T
            
            # 정규화
            points_rect = (R0_rect @ points_cam[:, :3].T).T
            
            # 카메라 좌표를 이미지 평면에 투영
            points_proj = (P2 @ np.vstack((points_rect.T, np.ones(len(points_rect))))).T
            points_proj = points_proj[:, :2] / points_proj[:, 2:]
            
            # 이미지 내부에 있는 포인트만 선택
            mask = (points_proj[:, 0] >= 0) & (points_proj[:, 0] < image.shape[1]) & \
                   (points_proj[:, 1] >= 0) & (points_proj[:, 1] < image.shape[0]) & \
                   (points_cam[:, 2] > 0)  # 카메라 앞쪽 포인트만
            
            points_proj = points_proj[mask]
            points_cam = points_cam[mask]
            filtered_points = filtered_points[mask]
            
            # 시각화
            plt.figure(figsize=(15, 5))
            
            # 원본 이미지
            plt.subplot(121)
            plt.imshow(image)
            plt.title(f'Original Image (Frame {frame_idx})', fontsize=12)
            plt.axis('off')
            
            # 라이다 포인트 오버레이
            plt.subplot(122)
            plt.imshow(image)
            
            # 거리에 따른 색상 매핑
            distances = np.linalg.norm(points_cam[:, :3], axis=1)
            colors = plt.cm.jet(distances / np.max(distances))
            
            # 포인트 크기는 거리에 반비례
            sizes = 100 / (distances + 1)
            
            plt.scatter(points_proj[:, 0], points_proj[:, 1], c=colors, s=sizes, alpha=0.6)
            plt.title(f'LiDAR Points Overlay (Frame {frame_idx})', fontsize=12)
            plt.axis('off')
            
            plt.tight_layout()
            
            # 결과 저장
            results_dir = Path('results') / self.date
            os.makedirs(results_dir, exist_ok=True)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f'kitti_camera_lidar_fusion_frame{frame_idx}_{timestamp}.png'
            filepath = results_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✅ 정합 시각화 결과가 저장되었습니다: {filepath}")
            
            plt.show()
            
            # 통계 출력
            print(f"\n📊 프레임 {frame_idx} 정합 통계:")
            print(f"  • 총 라이다 포인트: {len(filtered_points):,}")
            print(f"  • 이미지에 투영된 포인트: {len(points_proj):,}")
            print(f"  • 투영 비율: {len(points_proj)/len(filtered_points)*100:.1f}%")
            print(f"  • 최소 거리: {np.min(distances):.1f}m")
            print(f"  • 최대 거리: {np.max(distances):.1f}m")
            print(f"  • 평균 거리: {np.mean(distances):.1f}m")
            print("-" * 50)

def main():
    parser = argparse.ArgumentParser(description='KITTI Point Cloud Mapping')
    parser.add_argument('--base_path', type=str, 
                      default='KITTI_dataset',
                      help='KITTI 데이터셋의 기본 경로')
    parser.add_argument('--dataset_prefix', type=str,
                      default='2011_09_26_drive_0020',
                      help='데이터셋 프리픽스 (예: 2011_09_26_drive_0001)')
    parser.add_argument('--pose_file', type=str,
                      help='사전 계산된 pose 파일 경로 (기본값: results/{dataset_prefix}/kitti_poses_*.pkl)')
    parser.add_argument('--max_frames', type=int,
                      default=None,
                      help='처리할 최대 프레임 수 (기본값: 모든 프레임)')
    parser.add_argument('--voxel_size', type=float,
                      default=0.2,
                      help='포인트 클라우드 다운샘플링을 위한 복셀 크기')
    parser.add_argument('--output_dir', type=str,
                      default='results',
                      help='결과를 저장할 디렉토리')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    results_dir = Path(args.output_dir) / args.dataset_prefix
    os.makedirs(results_dir, exist_ok=True)
    
    # pose 파일 경로가 지정되지 않은 경우 기본값 사용
    if args.pose_file is None:
        pose_files = sorted(glob.glob(str(results_dir / 'kitti_poses_*.pkl')))
        if pose_files:
            args.pose_file = pose_files[-1]  # 가장 최근 파일 사용
            print(f"자동으로 최근 pose 파일을 사용합니다: {args.pose_file}")
        else:
            print("경고: pose 파일을 찾을 수 없습니다.")
    
    # Point Cloud Mapper 초기화
    mapper = ImprovedKITTIPointCloudMapper(args.base_path, args.dataset_prefix, args.pose_file)
    
    print("KITTI Point Cloud Mapping 시작...")
    print(f"데이터셋: {args.dataset_prefix}")
    
    # 테스트 맵 생성
    mapper.create_test_map_simple(max_frames=args.max_frames)
    
    # 최적화된 포인트 클라우드 생성 및 저장
    gt_points, estimated_points = mapper.create_optimized_point_clouds(voxel_size=args.voxel_size)
    
    if gt_points is not None and estimated_points is not None:
        # 결과 시각화
        mapper.visualize_comparison_2d()
        print(f"\n완료! 결과가 {results_dir} 디렉토리에 저장되었습니다.")
    else:
        print("\n❌ 포인트 클라우드 생성에 실패했습니다.")

if __name__ == "__main__":
    main() 