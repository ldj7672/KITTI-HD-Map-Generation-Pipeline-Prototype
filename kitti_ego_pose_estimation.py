import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.spatial.transform import Rotation
import os
import glob
from mpl_toolkits.mplot3d import Axes3D
import cv2
import open3d as o3d
from filterpy.kalman import KalmanFilter
from skimage.metrics import structural_similarity as ssim
import copy
import pickle
import json
from datetime import datetime
import argparse
from pathlib import Path

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'  # 영어 폰트로 설정
plt.rcParams['axes.unicode_minus'] = False

class KITTIEgoPoseEstimator:
    def __init__(self, base_path, dataset_prefix):
        self.base_path = Path(base_path)
        self.date = dataset_prefix
        self.drive = f'{dataset_prefix}_sync'
        
        # 데이터 경로 설정
        self.data_path = self.base_path / 'data' / self.drive
        self.calib_path = self.base_path / 'calibration' / self.date
        
        self.velodyne_path = self.data_path / 'velodyne_points' / 'data'
        self.oxts_path = self.data_path / 'oxts' / 'data'
        self.calib_cam_to_cam_path = self.calib_path / 'calib_cam_to_cam.txt'
        self.velo_to_cam_path = self.calib_path / 'calib_velo_to_cam.txt'
        
        # 파일 목록 가져오기
        self.velodyne_files = sorted(glob.glob(str(self.velodyne_path / '*.bin')))
        self.oxts_files = sorted(glob.glob(str(self.oxts_path / '*.txt')))
        
        # 캘리브레이션 데이터 로드
        self.calib = self.load_calibration()
        self.velo_to_cam = self.load_velo_to_cam_calibration()
        
        # 초기 위치 및 방향 설정
        self.initial_pose = None
        self.estimated_trajectory = []
        self.gt_trajectory = []
        
        # 이전 프레임 데이터 저장
        self.prev_points = None
        self.prev_imu_data = None
        
        # LIO 누적 pose 행렬
        self.lio_pose = np.eye(4)
        
        # 융합된 pose (LIO + GPS/IMU)
        self.fused_pose = np.eye(4)
        self.fused_trajectory = []
        
        # Extended Kalman Filter 초기화
        self.kf = self.init_extended_kalman_filter()
        
        # 스케일 팩터
        self.scale_factor = 1.0
        
    def init_extended_kalman_filter(self):
        """LIO + GPS/IMU 융합을 위한 Extended Kalman Filter 초기화"""
        kf = KalmanFilter(dim_x=9, dim_z=6)  # 상태: [x, y, z, vx, vy, vz, roll, pitch, yaw]
        
        # 상태 전이 행렬 (위치, 속도, 자세)
        dt = 0.1  # KITTI 샘플링 주기
        kf.F = np.eye(9)
        kf.F[0:3, 3:6] = dt * np.eye(3)  # 위치 = 위치 + 속도*dt
        
        # 측정 행렬 (GPS 위치 + LIO 위치)
        kf.H = np.zeros((6, 9))
        kf.H[0:3, 0:3] = np.eye(3)  # GPS 위치 측정
        kf.H[3:6, 0:3] = np.eye(3)  # LIO 위치 측정
        
        # 공분산 행렬 초기화
        kf.P *= 100
        kf.R = np.diag([1.0, 1.0, 1.0, 5.0, 5.0, 5.0])  # GPS 더 신뢰, LIO 덜 신뢰
        kf.Q = np.diag([0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 0.01, 0.01, 0.01])
        
        return kf
    
    def load_calibration(self):
        """캘리브레이션 파일 로드"""
        with open(self.calib_cam_to_cam_path, 'r') as f:
            lines = f.readlines()
        
        calib = {}
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                value = value.strip()
                
                if 'calib_time' in key:
                    continue
                    
                try:
                    calib[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    continue
        
        return calib
    
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
    
    def load_velo_to_cam_calibration(self):
        """라이다-카메라 변환 캘리브레이션 로드"""
        with open(self.velo_to_cam_path, 'r') as f:
            lines = f.readlines()
        
        calib = {}
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                value = value.strip()
                try:
                    calib[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    continue
        
        return calib
    
    def transform_points_to_camera_frame(self, points):
        """라이다 포인트를 카메라 좌표계로 변환"""
        # 라이다-카메라 변환 행렬
        R = self.velo_to_cam['R'].reshape(3, 3)
        T = self.velo_to_cam['T'].reshape(3, 1)
        
        # 포인트 변환
        points_cam = np.dot(points[:, :3], R.T) + T.T
        return points_cam
    
    def estimate_visual_odometry(self, curr_image, prev_image):
        """시각적 오도메트리 추정"""
        if prev_image is None:
            return np.eye(4)
        
        # 특징점 검출 및 매칭
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(prev_image, None)
        kp2, des2 = sift.detectAndCompute(curr_image, None)
        
        # 특징점 매칭
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        
        # 좋은 매칭 선택
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        
        if len(good_matches) < 8:
            return np.eye(4)
        
        # 매칭된 점들의 좌표 추출
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Essential 행렬 계산
        E, mask = cv2.findEssentialMat(src_pts, dst_pts, focal=1.0, pp=(0., 0.))
        
        # 회전과 이동 추출
        _, R, t, _ = cv2.recoverPose(E, src_pts, dst_pts)
        
        # 변환 행렬 생성
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.ravel()
        
        return T
    
    def estimate_lidar_odometry(self, curr_points, prev_points):
        """라이다 오도메트리 추정"""
        if prev_points is None:
            return np.eye(4)
        
        # 원본 라이다 좌표계에서 직접 ICP 수행 (좌표계 변환 제거)
        # Open3D 포인트 클라우드 변환
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(prev_points[:, :3])  # 원본 라이다 좌표계 사용
        
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(curr_points[:, :3])  # 원본 라이다 좌표계 사용
        
        # 포인트 클라우드 전처리 (덜 과도하게)
        # 1. 노이즈 제거 (더 관대하게)
        pcd1, _ = pcd1.remove_statistical_outlier(nb_neighbors=10, std_ratio=3.0)
        pcd2, _ = pcd2.remove_statistical_outlier(nb_neighbors=10, std_ratio=3.0)
        
        # 2. 다운샘플링 (더 큰 복셀 사이즈)
        voxel_size = 0.3  # 0.05에서 0.3으로 증가
        pcd1 = pcd1.voxel_down_sample(voxel_size)
        pcd2 = pcd2.voxel_down_sample(voxel_size)
        
        # 3. 정규화 제거 (실제 이동량 보존)
        # pcd1_center = pcd1.get_center()
        # pcd2_center = pcd2.get_center()
        # pcd1.translate(-pcd1_center)
        # pcd2.translate(-pcd2_center)
        
        # ICP 알고리즘으로 정합 (더 관대한 파라미터)
        result = o3d.pipelines.registration.registration_icp(
            pcd1, pcd2, 
            max_correspondence_distance=2.0,  # 0.3에서 2.0으로 증가
            init=np.eye(4),
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
        )
        
        # 변환 행렬 복사
        transformation = result.transformation.copy()
        
        # 이동량 확인 및 출력
        translation = transformation[:3, 3]
        translation_norm = np.linalg.norm(translation)
        
        # 디버깅 출력
        if translation_norm > 0.001:  # 의미있는 이동이 있을 때만 출력
            print(f"ICP 이동량: {translation}, 거리: {translation_norm:.3f}m")
        
        # 스케일 팩터 자동 조정 (GT 기반)
        # 첫 10프레임 동안 GT와 비교하여 스케일 조정
        if hasattr(self, 'frame_count'):
            self.frame_count += 1
        else:
            self.frame_count = 1
            
        if self.frame_count <= 10 and translation_norm > 0.001:
            # GT와 현재 ICP 결과를 비교하여 스케일 추정
            # 이는 간단한 휴리스틱이며, 실제로는 더 정교한 방법이 필요할 수 있음
            expected_movement = 1.0  # KITTI에서 프레임당 예상 이동량 (약 1m)
            if translation_norm > 0:
                estimated_scale = expected_movement / translation_norm
                # 스케일 팩터를 점진적으로 업데이트
                self.scale_factor = 0.9 * self.scale_factor + 0.1 * estimated_scale
                print(f"스케일 팩터 업데이트: {self.scale_factor:.3f}")
        
        # 스케일 조정 적용
        transformation[:3, 3] *= self.scale_factor
        
        return transformation
    
    def estimate_ego_pose(self, frame_idx):
        """LIO + GPS/IMU 융합으로 ego-pose 추정"""
        # 현재 프레임 데이터 로드
        curr_points = self.load_velodyne_points(self.velodyne_files[frame_idx])
        curr_imu_data = self.load_oxts_data(self.oxts_files[frame_idx])
        
        # LIO 추정 (순수 라이다 기반)
        if self.prev_points is None:
            self.prev_points = curr_points
            self.prev_imu_data = curr_imu_data
            lio_position = self.lio_pose[:3, 3]
            lio_rotation = Rotation.from_matrix(self.lio_pose[:3, :3])
        else:
            T_lio = self.estimate_lidar_odometry(curr_points, self.prev_points)
            self.lio_pose = self.lio_pose @ T_lio
            lio_position = self.lio_pose[:3, 3]
            lio_rotation = Rotation.from_matrix(self.lio_pose[:3, :3])
        
        # GPS 절대 위치 계산
        if self.initial_pose is None:
            self.initial_pose = {
                'lat': curr_imu_data['lat'],
                'lon': curr_imu_data['lon'],
                'alt': curr_imu_data['alt']
            }
        
        gps_position = self.lat_lon_to_xyz(
            curr_imu_data['lat'], curr_imu_data['lon'], curr_imu_data['alt'],
            self.initial_pose['lat'], self.initial_pose['lon'], self.initial_pose['alt']
        )
        
        # EKF 예측 단계
        self.kf.predict()
        
        # IMU 데이터로 상태 업데이트
        if self.prev_imu_data is not None:
            # 속도 계산 (IMU)
            imu_velocity = np.array([curr_imu_data['vn'], curr_imu_data['ve'], curr_imu_data['vf']])
            # 상태에 IMU 속도 반영 (shape 맞춤)
            self.kf.x[3:6, 0] = imu_velocity  # (3,1) shape으로 할당
        
        # 측정값 (GPS 위치 + LIO 위치)
        measurement = np.concatenate([gps_position, lio_position])
        
        # EKF 업데이트 단계
        self.kf.update(measurement)
        
        # 융합된 결과
        fused_position = self.kf.x[:3, 0].copy()  # (3,1) -> (3,) 변환
        
        # IMU 회전 정보 사용
        fused_rotation = Rotation.from_euler('xyz', [
            curr_imu_data['roll'],
            curr_imu_data['pitch'], 
            curr_imu_data['yaw']
        ])
        
        # 융합된 pose 업데이트
        self.fused_pose[:3, :3] = fused_rotation.as_matrix()
        self.fused_pose[:3, 3] = fused_position
        
        # 현재 상태 저장
        self.prev_points = curr_points
        self.prev_imu_data = curr_imu_data
        
        return fused_position, fused_rotation
    
    def get_gt_pose(self, frame_idx):
        """Ground Truth pose 가져오기"""
        oxts_data = self.load_oxts_data(self.oxts_files[frame_idx])
        
        if self.initial_pose is None:
            self.initial_pose = {
                'lat': oxts_data['lat'],
                'lon': oxts_data['lon'],
                'alt': oxts_data['alt']
            }
        
        position = self.lat_lon_to_xyz(
            oxts_data['lat'], oxts_data['lon'], oxts_data['alt'],
            self.initial_pose['lat'], self.initial_pose['lon'], self.initial_pose['alt']
        )
        
        rotation = Rotation.from_euler('xyz', [
            oxts_data['roll'],
            oxts_data['pitch'],
            oxts_data['yaw']
        ])
        
        return position, rotation
    
    def lat_lon_to_xyz(self, lat, lon, alt, ref_lat, ref_lon, ref_alt):
        """위도/경도를 XYZ 좌표로 변환"""
        x = (lon - ref_lon) * 111320 * np.cos(np.radians(ref_lat))
        y = (lat - ref_lat) * 111320
        z = alt - ref_alt
        return np.array([x, y, z])
    
    def process_sequence(self, start_idx=0, end_idx=None):
        """시퀀스 전체 처리 (LIO + GPS/IMU 융합)"""
        if end_idx is None:
            end_idx = len(self.velodyne_files)
        
        # 실제 파일 수에 맞춰 end_idx 제한
        max_available_frames = min(len(self.velodyne_files), len(self.oxts_files))
        end_idx = min(end_idx, max_available_frames)
        
        print(f"처리할 프레임 범위: {start_idx} ~ {end_idx-1}")
        print(f"사용 가능한 velodyne 파일: {len(self.velodyne_files)}개")
        print(f"사용 가능한 oxts 파일: {len(self.oxts_files)}개")
        print(f"실제 처리할 프레임 수: {end_idx - start_idx}개")
        
        for frame_idx in range(start_idx, end_idx):
            # 융합된 pose 추정
            fused_position, fused_rotation = self.estimate_ego_pose(frame_idx)
            self.estimated_trajectory.append(fused_position)
            
            # GT pose
            gt_position, gt_rotation = self.get_gt_pose(frame_idx)
            self.gt_trajectory.append(gt_position)
            
            # 진행 상황 출력 (매 10프레임마다)
            if frame_idx % 10 == 0:
                print(f"프레임 {frame_idx} 처리 중...")
                print(f"융합된 위치: {fused_position.reshape(-1)}")
                print(f"GT 위치: {gt_position.reshape(-1)}")
                print(f"위치 오차: {np.linalg.norm(fused_position.reshape(-1) - gt_position.reshape(-1)):.3f} m")
                print("-" * 50)
    
    def calculate_trajectory_error(self):
        """추정된 trajectory와 GT trajectory 간의 오차 계산"""
        estimated_traj = np.array(self.estimated_trajectory).reshape(-1, 3)  # shape을 (N,3)으로 변환
        gt_traj = np.array(self.gt_trajectory)
        
        # 위치 오차 계산
        position_errors = np.linalg.norm(estimated_traj - gt_traj, axis=1)
        
        # 오차 메트릭 계산
        mse = np.mean(position_errors ** 2)
        mae = np.mean(position_errors)
        max_error = np.max(position_errors)
        
        return {
            'mse': mse,
            'mae': mae,
            'max_error': max_error,
            'position_errors': position_errors
        }
    
    def visualize_trajectories(self):
        """XY 평면 trajectory 시각화 및 오차 리포팅"""
        estimated_traj = np.array(self.estimated_trajectory).reshape(-1, 3)
        gt_trajectory = np.array(self.gt_trajectory)
        
        # 오차 계산
        error_metrics = self.calculate_trajectory_error()
        
        # 스타일 설정
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # 색상 설정
        color_estimated = '#2E86AB'  # 파란색
        color_gt = '#A23B72'        # 자주색
        
        # XY 평면 궤적 시각화
        ax.plot(estimated_traj[:, 0], estimated_traj[:, 1], 
                color=color_estimated, linewidth=3, label='LIO+GPS/IMU Fusion', 
                marker='o', markersize=3, alpha=0.8)
        ax.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], 
                color=color_gt, linewidth=3, linestyle='--', label='Ground Truth', 
                marker='s', markersize=3, alpha=0.8)
        
        # 시작점과 끝점 표시
        ax.scatter(estimated_traj[0, 0], estimated_traj[0, 1], 
                   color='green', s=150, marker='o', label='Start Point', zorder=5, edgecolor='white', linewidth=2)
        ax.scatter(estimated_traj[-1, 0], estimated_traj[-1, 1], 
                   color='red', s=150, marker='X', label='End Point', zorder=5)
        
        # 몇 개 프레임마다 방향 화살표 표시
        step = max(1, len(estimated_traj) // 10)
        for i in range(0, len(estimated_traj)-1, step):
            dx = estimated_traj[i+1, 0] - estimated_traj[i, 0]
            dy = estimated_traj[i+1, 1] - estimated_traj[i, 1]
            if abs(dx) > 0.1 or abs(dy) > 0.1:  # 의미있는 이동이 있을 때만
                ax.arrow(estimated_traj[i, 0], estimated_traj[i, 1], dx*0.5, dy*0.5,
                        head_width=1.5, head_length=1.0, fc=color_estimated, ec=color_estimated, alpha=0.6)
        
        # 축 설정
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=13, loc='best', frameon=True, fancybox=True, shadow=True)
        ax.set_title('Vehicle Trajectory Comparison (XY Plane)', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('X Position (m)', fontsize=14)
        ax.set_ylabel('Y Position (m)', fontsize=14)
        
        # 축 눈금 크기 조정
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        plt.tight_layout(pad=2.0)
        
        # results 폴더 생성
        results_dir = Path('results') / self.date
        os.makedirs(results_dir, exist_ok=True)
        
        # 그림 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'kitti_trajectory_{timestamp}.png'
        filepath = results_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\n궤적 그림이 저장되었습니다: {filepath}")
        
        plt.show()
        
        # 상세한 오차 리포팅
        self.print_error_report(error_metrics)
    
    def print_error_report(self, error_metrics):
        """상세한 오차 분석 리포트 출력"""
        print("\n" + "="*60)
        print("               KITTI EGO-POSE ESTIMATION REPORT")
        print("="*60)
        print(f"Algorithm: LIO (ICP-based) + GPS/IMU Sensor Fusion")
        print(f"Dataset: KITTI (Frames: {len(error_metrics['position_errors'])})")
        print(f"Processing Time: ~{len(error_metrics['position_errors']) * 0.1:.1f} seconds of driving")
        print("-"*60)
        
        print("TRAJECTORY ERROR ANALYSIS:")
        print(f"  • Mean Squared Error (MSE):     {error_metrics['mse']:.3f} m²")
        print(f"  • Mean Absolute Error (MAE):    {error_metrics['mae']:.3f} m")
        print(f"  • Root Mean Square Error (RMSE): {np.sqrt(error_metrics['mse']):.3f} m")
        print(f"  • Maximum Error:                {error_metrics['max_error']:.3f} m")
        print(f"  • Minimum Error:                {np.min(error_metrics['position_errors']):.3f} m")
        print(f"  • Standard Deviation:           {np.std(error_metrics['position_errors']):.3f} m")
        
        print("-"*60)
        print("PERFORMANCE EVALUATION:")
        
        # 성능 등급 평가
        mae = error_metrics['mae']
        if mae < 2.0:
            grade = "EXCELLENT"
            description = "Commercial SLAM system level"
        elif mae < 5.0:
            grade = "VERY GOOD"
            description = "Research-grade SLAM system level"
        elif mae < 10.0:
            grade = "GOOD"
            description = "Acceptable for navigation applications"
        elif mae < 20.0:
            grade = "FAIR"
            description = "Needs improvement for practical use"
        else:
            grade = "POOR"
            description = "Significant improvements required"
        
        print(f"  • Performance Grade:            {grade}")
        print(f"  • Assessment:                   {description}")
        print(f"  • Accuracy Percentage:          {max(0, 100 - mae/1.0):.1f}%")
        
        # 에러 분포 분석
        errors = error_metrics['position_errors']
        percentile_50 = np.percentile(errors, 50)
        percentile_75 = np.percentile(errors, 75)
        percentile_95 = np.percentile(errors, 95)
        
        print("-"*60)
        print("ERROR DISTRIBUTION:")
        print(f"  • 50% of errors are below:      {percentile_50:.3f} m")
        print(f"  • 75% of errors are below:      {percentile_75:.3f} m") 
        print(f"  • 95% of errors are below:      {percentile_95:.3f} m")
        
        print("-"*60)
        print("SENSOR FUSION BENEFITS:")
        print("  • GPS: Provides absolute position reference")
        print("  • IMU: Contributes velocity and orientation data")
        print("  • LiDAR: Delivers precise relative motion estimation")
        print("  • Fusion: Reduces cumulative drift significantly")
        
        print("="*60)
        print("Report generated successfully!")
        print("="*60)

    def save_poses_to_file(self, output_dir):
        """계산된 pose들을 파일로 저장"""
        # 결과 디렉토리 생성
        results_dir = Path(output_dir) / self.date
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'kitti_poses_{timestamp}.pkl'
        filepath = results_dir / filename
        
        # 저장할 데이터 구성
        pose_data = {
            'estimated_trajectory': self.estimated_trajectory,
            'gt_trajectory': self.gt_trajectory,
            'fused_poses': [],  # 4x4 변환 행렬들
            'gt_poses': [],     # 4x4 변환 행렬들
            'metadata': {
                'dataset_prefix': self.date,
                'total_frames': len(self.estimated_trajectory),
                'algorithm': 'LIO + GPS/IMU Fusion',
                'dataset': 'KITTI',
                'timestamp': datetime.now().isoformat()
            }
        }
        
        # 4x4 변환 행렬 형태로 pose 저장
        for i, (est_pos, gt_pos) in enumerate(zip(self.estimated_trajectory, self.gt_trajectory)):
            # 추정된 pose
            est_pose_matrix = np.eye(4)
            est_pose_matrix[:3, 3] = est_pos.reshape(-1)  # 위치
            
            # IMU에서 회전 정보 가져오기
            oxts_data = self.load_oxts_data(self.oxts_files[i])
            est_rotation = Rotation.from_euler('xyz', [
                oxts_data['roll'], oxts_data['pitch'], oxts_data['yaw']
            ])
            est_pose_matrix[:3, :3] = est_rotation.as_matrix()  # 회전 행렬 추가
            pose_data['fused_poses'].append(est_pose_matrix)
            
            # GT pose (oxts에서 회전 정보도 포함)
            gt_rotation = Rotation.from_euler('xyz', [
                oxts_data['roll'], oxts_data['pitch'], oxts_data['yaw']
            ])
            gt_pose_matrix = np.eye(4)
            gt_pose_matrix[:3, :3] = gt_rotation.as_matrix()
            gt_pose_matrix[:3, 3] = gt_pos.reshape(-1)
            pose_data['gt_poses'].append(gt_pose_matrix)
        
        # 파일로 저장
        with open(filepath, 'wb') as f:
            pickle.dump(pose_data, f)
        
        print(f"\nPose 데이터가 저장되었습니다: {filepath}")
        print(f"  • 데이터셋: {pose_data['metadata']['dataset_prefix']}")
        print(f"  • 총 프레임 수: {pose_data['metadata']['total_frames']}")
        print(f"  • 알고리즘: {pose_data['metadata']['algorithm']}")
        
        # JSON 형태로도 저장 (가독성을 위해)
        json_filename = filename.replace('.pkl', '.json')
        json_filepath = results_dir / json_filename
        json_data = {
            'metadata': pose_data['metadata'],
            'estimated_positions': [pos.tolist() for pos in self.estimated_trajectory],
            'gt_positions': [pos.tolist() for pos in self.gt_trajectory],
            'estimated_rotations': [[oxts_data['roll'], oxts_data['pitch'], oxts_data['yaw']] 
                                  for oxts_data in [self.load_oxts_data(f) for f in self.oxts_files[:len(self.estimated_trajectory)]]],
            'gt_rotations': [[oxts_data['roll'], oxts_data['pitch'], oxts_data['yaw']] 
                           for oxts_data in [self.load_oxts_data(f) for f in self.oxts_files[:len(self.gt_trajectory)]]]
        }
        
        with open(json_filepath, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"  • JSON 파일도 저장됨: {json_filepath}")
        
        return filepath

def main():
    parser = argparse.ArgumentParser(description='KITTI Ego-Pose Estimation')
    parser.add_argument('--base_path', type=str, 
                      default='KITTI_dataset',
                      help='KITTI 데이터셋의 기본 경로')
    parser.add_argument('--dataset_prefix', type=str,
                      default='2011_09_26_drive_0001',
                      help='데이터셋 프리픽스 (예: 2011_09_26_drive_0001)')
    parser.add_argument('--max_frames', type=int,
                      default=100,
                      help='처리할 최대 프레임 수')
    parser.add_argument('--output_dir', type=str,
                      default='results',
                      help='결과를 저장할 디렉토리')
    parser.add_argument('--no_save_poses', action='store_true',
                      help='계산된 pose를 파일로 저장하지 않을지 여부')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Ego-pose 추정기 초기화
    estimator = KITTIEgoPoseEstimator(args.base_path, args.dataset_prefix)
    
    print("KITTI Ego-Pose Estimation 시작...")
    print(f"데이터셋: {args.dataset_prefix}")
    
    # 시퀀스 처리
    estimator.process_sequence(0, args.max_frames)
    
    # Pose 결과를 파일로 저장 (기본적으로 저장)
    if not args.no_save_poses:
        pose_filename = estimator.save_poses_to_file(args.output_dir)
        print(f"\nPose 파일이 저장되었습니다: {pose_filename}")
    
    # 결과 시각화
    estimator.visualize_trajectories()
    
    print(f"\n완료! 결과가 {args.output_dir}/{args.dataset_prefix} 디렉토리에 저장되었습니다.")

if __name__ == "__main__":
    main() 