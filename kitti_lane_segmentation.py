import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2
import time
import pickle
from scipy.spatial.transform import Rotation
import open3d as o3d
import glob
import argparse
from pathlib import Path

class KITTILaneSegmentation:
    def __init__(self, base_path, dataset_prefix, pose_file=None):
        self.base_path = Path(base_path)
        self.date = dataset_prefix
        self.drive = f'{dataset_prefix}_sync'
        
        # 데이터 경로 설정
        self.data_path = self.base_path / 'data' / self.drive
        self.calib_path = self.base_path / 'calibration' / self.date
        
        self.image_path = self.data_path / 'image_02' / 'data'
        self.velodyne_path = self.data_path / 'velodyne_points' / 'data'
        
        # 파일 목록 가져오기
        self.image_files = sorted(glob.glob(str(self.image_path / '*.png')))
        self.velodyne_files = sorted(glob.glob(str(self.velodyne_path / '*.bin')))
        
        # 사전 계산된 pose 로드
        self.pose_file = pose_file
        self.precomputed_poses = None
        if pose_file:
            self.load_precomputed_poses(pose_file)
        
        # 캘리브레이션 데이터 로드
        self.load_calibration()
        
        # 이미지 크기는 첫 번째 이미지를 읽어서 확인
        if self.image_files:
            sample_image = cv2.imread(self.image_files[0])
            self.image_height, self.image_width = sample_image.shape[:2]
            print(f"이미지 크기: {self.image_width} x {self.image_height}")
        else:
            # 기본값 (KITTI 표준 크기)
            self.image_width, self.image_height = 1242, 375
            
        # ROI 설정 (이미지 하단 절반)
        half_height = self.image_height // 2
        self.roi_vertices = np.array([
            [(0, self.image_height), (self.image_width, self.image_height), 
             (self.image_width, half_height), (0, half_height)]
        ], dtype=np.int32)
        
        print(f"ROI 설정: y={half_height} ~ {self.image_height} (하단 절반)")
    
    def load_calibration(self):
        """캘리브레이션 데이터 로드"""
        try:
            # 카메라 캘리브레이션
            calib_path = self.calib_path / 'calib_cam_to_cam.txt'
            with open(calib_path, 'r') as f:
                calib_data = f.readlines()
            
            # P2 (카메라 프로젝션 행렬)
            P2_line = [line for line in calib_data if line.startswith('P_rect_02')][0]
            self.P2 = np.array([float(x) for x in P2_line.strip().split()[1:13]]).reshape(3, 4)
            
            # R0_rect (정규화 행렬)
            R0_line = [line for line in calib_data if line.startswith('R_rect_02')][0]
            self.R0_rect = np.array([float(x) for x in R0_line.strip().split()[1:10]]).reshape(3, 3)
            
            # 라이다->카메라 변환 행렬
            velo_to_cam_path = self.calib_path / 'calib_velo_to_cam.txt'
            with open(velo_to_cam_path, 'r') as f:
                velo_to_cam_data = f.readlines()
            
            R_line = [line for line in velo_to_cam_data if line.startswith('R:')][0]
            self.R = np.array([float(x) for x in R_line.strip().split()[1:10]]).reshape(3, 3)
            
            T_line = [line for line in velo_to_cam_data if line.startswith('T:')][0]
            self.T = np.array([float(x) for x in T_line.strip().split()[1:4]]).reshape(3, 1)
            
            self.Tr_velo_to_cam = np.hstack((self.R, self.T))
            
            print("✅ 캘리브레이션 데이터 로드 성공")
        except Exception as e:
            print(f"❌ 캘리브레이션 데이터 로드 실패: {e}")
    
    def load_precomputed_poses(self, pose_file):
        """사전 계산된 pose 파일 로드"""
        try:
            with open(pose_file, 'rb') as f:
                self.precomputed_poses = pickle.load(f)
            print(f"✅ Pose 파일 로드 성공: {pose_file}")
        except Exception as e:
            print(f"❌ Pose 파일 로드 실패: {e}")
    
    def detect_white_lanes(self, image):
        """밝은 아스팔트에서 하얀색 차선 검출"""
        # 1. 그레이스케일 변환
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 2. 밝은 아스팔트를 고려한 높은 임계값 사용
        # 매우 밝은 픽셀만 차선으로 간주
        white_mask1 = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]  # 높은 임계값
        white_mask2 = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)[1]  # 매우 높은 임계값
        white_mask3 = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)[1]  # 극도로 높은 임계값
        
        # 모든 마스크 결합
        white_mask = cv2.bitwise_or(white_mask1, white_mask2)
        white_mask = cv2.bitwise_or(white_mask, white_mask3)
        
        # 3. ROI 적용 (이미지 하단 절반) - 동적 크기
        roi_mask = np.zeros_like(white_mask)
        cv2.fillPoly(roi_mask, self.roi_vertices, 255)
        white_mask_roi = cv2.bitwise_and(white_mask, roi_mask)
        
        # 4. 가벼운 노이즈 제거
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        white_mask_roi = cv2.morphologyEx(white_mask_roi, cv2.MORPH_OPEN, kernel_small)
        white_mask_roi = cv2.morphologyEx(white_mask_roi, cv2.MORPH_CLOSE, kernel_small)
        
        # 5. 엣지 검출
        # 원본 그레이스케일에서 직접 엣지 검출 (ROI 적용)
        gray_roi = cv2.bitwise_and(gray, roi_mask)
        gray_blur = cv2.GaussianBlur(gray_roi, (5, 5), 0)
        edges = cv2.Canny(gray_blur, 100, 200)  # 높은 임계값으로 확실한 엣지만
        
        # 마스크 기반 엣지도 추가
        mask_blur = cv2.GaussianBlur(white_mask_roi, (3, 3), 0)
        mask_edges = cv2.Canny(mask_blur, 50, 150)
        
        # 두 엣지 결합
        final_edges = cv2.bitwise_or(edges, mask_edges)
        
        # 디버깅 정보 출력
        white_pixels = np.sum(white_mask > 0)
        roi_white_pixels = np.sum(white_mask_roi > 0)
        edge_pixels = np.sum(final_edges > 0)
        
        # ROI 영역의 평균 밝기도 출력
        roi_area = gray[roi_mask > 0]
        avg_brightness = np.mean(roi_area) if len(roi_area) > 0 else 0
        
        print(f"  디버깅: ROI 평균밝기={avg_brightness:.1f}, 흰색 픽셀={white_pixels}, ROI 흰색 픽셀={roi_white_pixels}, 엣지 픽셀={edge_pixels}")
        
        return white_mask, white_mask_roi, final_edges
    
    def extract_lane_lines(self, edges):
        """차선 라인 추출 (매우 관대한 설정)"""
        # 허프 변환으로 직선 검출 (매우 관대한 파라미터)
        lines = cv2.HoughLinesP(
            edges, 
            rho=1, 
            theta=np.pi/180, 
            threshold=10,      # 매우 낮은 임계값
            minLineLength=20,  # 매우 짧은 라인도 허용
            maxLineGap=50      # 큰 갭도 허용
        )
        
        # 디버깅 정보
        if lines is not None:
            print(f"  디버깅: 허프 변환으로 {len(lines)}개 라인 검출")
            
            # 검출된 라인들의 기울기 분포 확인
            slopes = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 - x1 != 0:
                    slope = (y2 - y1) / (x2 - x1)
                    slopes.append(slope)
            
            if slopes:
                print(f"  디버깅: 기울기 범위 = {min(slopes):.2f} ~ {max(slopes):.2f}")
        else:
            print(f"  디버깅: 허프 변환으로 라인을 찾지 못함")
        
        # 라인을 찾지 못한 경우 빈 리스트 반환
        if lines is None:
            return [], []
        
        # 기울기로 좌측/우측 차선 분류 (매우 관대한 범위)
        left_lines = []
        right_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:  # 수직선 제외
                continue
            slope = (y2 - y1) / (x2 - x1)
            
            # 기울기 필터링 (매우 관대하게 - 거의 모든 기울기 허용)
            if abs(slope) < 0.1:  # 너무 수평인 것만 제외
                continue
                
            if slope < 0:  # 좌측 차선 (음의 기울기)
                left_lines.append(line[0])
            else:  # 우측 차선 (양의 기울기)
                right_lines.append(line[0])
        
        print(f"  디버깅: 좌측 차선 {len(left_lines)}개, 우측 차선 {len(right_lines)}개 검출")
        
        # 기울기별로 라인들 출력 (처음 5개만)
        if left_lines:
            print(f"  좌측 차선 예시: {left_lines[:3]}")
        if right_lines:
            print(f"  우측 차선 예시: {right_lines[:3]}")
        
        return left_lines, right_lines
    
    def fit_lane_lines(self, lines, image_shape):
        """차선 라인 피팅"""
        if len(lines) == 0:
            return None
        
        # 모든 점들을 하나의 배열로 합치기
        x_coords = []
        y_coords = []
        
        for line in lines:
            x1, y1, x2, y2 = line
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])
        
        # 다항식 피팅 (1차)
        if len(x_coords) >= 2:
            coeffs = np.polyfit(y_coords, x_coords, 1)
            
            # y 좌표 범위 설정
            y_start = int(image_shape[0] * 0.6)  # 이미지 하단 40%
            y_end = image_shape[0]
            
            # 피팅된 라인의 시작점과 끝점 계산
            x_start = int(coeffs[0] * y_start + coeffs[1])
            x_end = int(coeffs[0] * y_end + coeffs[1])
            
            return [(x_start, y_start, x_end, y_end)]
        
        return None
    
    def segment_lanes(self, image):
        """차선 검출 및 분할"""
        # 1. 하얀색 차선 검출
        white_mask, white_mask_roi, edges = self.detect_white_lanes(image)
        
        # 2. 차선 라인 추출
        left_lines, right_lines = self.extract_lane_lines(edges)
        
        # 3. 차선 라인 피팅
        fitted_left = self.fit_lane_lines(left_lines, image.shape)
        fitted_right = self.fit_lane_lines(right_lines, image.shape)
        
        # 4. 최종 차선 마스크 생성
        lane_mask = np.zeros_like(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
        
        # 검출된 라인들 그리기
        all_lines = []
        if fitted_left:
            all_lines.extend(fitted_left)
            for line in fitted_left:
                x1, y1, x2, y2 = line
                cv2.line(lane_mask, (x1, y1), (x2, y2), 255, 3)
        
        if fitted_right:
            all_lines.extend(fitted_right)
            for line in fitted_right:
                x1, y1, x2, y2 = line
                cv2.line(lane_mask, (x1, y1), (x2, y2), 255, 3)
        
        return white_mask, lane_mask, all_lines
    
    def extract_lane_boundaries(self, lane_mask, lines):
        """차선 경계선 추출"""
        # 스켈레톤 추출
        if np.any(lane_mask):
            skeleton = cv2.ximgproc.thinning(lane_mask)
        else:
            skeleton = lane_mask.copy()
        
        return skeleton, lines

    def visualize_lane_segmentation(self, frame_idx, show_lidar=True):
        """프레임별 차선 검출 결과 시각화"""
        # 1. 이미지 로드
        image_path = self.image_path / f'{frame_idx:010d}.png'
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 2. 차선 검출 (원시 결과만)
        white_mask, white_mask_roi, edges = self.detect_white_lanes(image)
        
        # 3. 시각화 - 프레임별 차선 검출 결과
        plt.figure(figsize=(15, 10))
        
        # 3.1 원본 이미지 (ROI 표시)
        plt.subplot(221)
        plt.imshow(image)
        roi_overlay = image.copy()
        cv2.polylines(roi_overlay, self.roi_vertices, True, (255, 0, 0), 3)
        plt.imshow(roi_overlay)
        plt.title('Original Image with ROI', fontsize=14)
        plt.axis('off')
        
        # 3.2 밝기 임계값 결과
        plt.subplot(222)
        plt.imshow(white_mask, cmap='gray')
        plt.title('Brightness Threshold', fontsize=14)
        plt.axis('off')
        
        # 3.3 ROI 적용 결과
        plt.subplot(223)
        plt.imshow(white_mask_roi, cmap='gray')
        plt.title('ROI Applied', fontsize=14)
        plt.axis('off')
        
        # 3.4 원본에 차선 오버레이
        plt.subplot(224)
        plt.imshow(image)
        
        # 차선 검출 결과를 노란색으로 오버레이
        lane_overlay = np.zeros_like(image)
        lane_overlay[white_mask_roi > 0] = [255, 255, 0]  # 노란색
        
        # 반투명 오버레이
        result = cv2.addWeighted(image, 0.7, lane_overlay, 0.3, 0)
        plt.imshow(result)
        plt.title('Lane Detection Overlay', fontsize=14)
        plt.axis('off')
        
        plt.tight_layout()
        
        # 4. 결과 저장
        results_dir = Path('results_lane') / self.date
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f'kitti_lane_detection_frame{frame_idx}_{timestamp}.png'
        filepath = results_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✅ 프레임 {frame_idx} 차선 검출 결과가 저장되었습니다: {filepath}")
        
        # 검출 통계 출력
        white_pixels = np.sum(white_mask > 0)
        roi_white_pixels = np.sum(white_mask_roi > 0)
        edge_pixels = np.sum(edges > 0)
        
        print(f"프레임 {frame_idx} 검출 통계:")
        print(f"  - 밝기 기반 흰색 픽셀: {white_pixels}개")
        print(f"  - ROI 적용 흰색 픽셀: {roi_white_pixels}개")
        print(f"  - 엣지 픽셀: {edge_pixels}개")
        
        # 5. 라이다 포인트와의 융합 (선택적)
        if show_lidar:
            self.visualize_lidar_fusion(frame_idx, white_mask_roi)
    
    def visualize_lidar_fusion(self, frame_idx, lane_mask):
        """라이다 포인트와 차선 검출 결과 융합 시각화"""
        # 1. 이미지 로드
        image_path = self.image_path / f'{frame_idx:010d}.png'
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 2. 라이다 포인트 로드
        points = np.fromfile(self.velodyne_files[frame_idx], dtype=np.float32).reshape(-1, 4)
        
        # 3. 포인트 클라우드를 카메라 좌표계로 변환
        points_homo = np.ones((len(points), 4))
        points_homo[:, :3] = points[:, :3]
        
        # 라이다 -> 카메라 변환
        points_cam = (self.Tr_velo_to_cam @ points_homo.T).T
        
        # 정규화
        points_rect = (self.R0_rect @ points_cam[:, :3].T).T
        
        # 카메라 좌표를 이미지 평면에 투영
        points_proj = (self.P2 @ np.vstack((points_rect.T, np.ones(len(points_rect))))).T
        points_proj = points_proj[:, :2] / points_proj[:, 2:]
        
        # 이미지 내부에 있는 포인트만 선택
        mask = (points_proj[:, 0] >= 0) & (points_proj[:, 0] < image.shape[1]) & \
               (points_proj[:, 1] >= 0) & (points_proj[:, 1] < image.shape[0]) & \
               (points_cam[:, 2] > 0)
        
        points_proj = points_proj[mask]
        points_cam = points_cam[mask]
        
        # 4. 시각화
        plt.figure(figsize=(15, 5))
        
        # 4.1 원본 이미지
        plt.subplot(131)
        plt.imshow(image)
        plt.title('Original Image', fontsize=14)
        plt.axis('off')
        
        # 4.2 라이다 포인트 오버레이
        plt.subplot(132)
        plt.imshow(image)
        
        # 거리에 따른 색상 매핑
        distances = np.linalg.norm(points_cam[:, :3], axis=1)
        colors = plt.cm.jet(distances / np.max(distances))
        
        # 포인트 크기는 거리에 반비례
        sizes = 30 / (distances + 1)
        
        plt.scatter(points_proj[:, 0], points_proj[:, 1], c=colors, s=sizes, alpha=0.6)
        plt.title('LiDAR Points', fontsize=14)
        plt.axis('off')
        
        # 4.3 차선과 라이다 포인트 융합
        plt.subplot(133)
        plt.imshow(image)
        
        # 차선 검출 결과를 노란색으로 오버레이
        lane_overlay = np.zeros_like(image)
        lane_overlay[lane_mask > 0] = [255, 255, 0]  # 노란색
        
        # 반투명 오버레이
        result = cv2.addWeighted(image, 0.6, lane_overlay, 0.4, 0)
        plt.imshow(result)
        
        # 라이다 포인트 오버레이
        plt.scatter(points_proj[:, 0], points_proj[:, 1], c=colors, s=sizes, alpha=0.5)
        plt.title('Lane + LiDAR Fusion', fontsize=14)
        plt.axis('off')
        
        plt.tight_layout()
        
        # 5. 결과 저장
        results_dir = Path('results') / self.date
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f'kitti_lidar_fusion_frame{frame_idx}_{timestamp}.png'
        filepath = results_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✅ 프레임 {frame_idx} 라이다 융합 결과가 저장되었습니다: {filepath}")

    def get_pose_from_precomputed(self, frame_idx, pose_type='gt'):
        """사전 계산된 pose에서 특정 프레임의 pose를 가져옴"""
        if self.precomputed_poses is None:
            return None
            
        try:
            if pose_type == 'gt':
                return self.precomputed_poses['gt_poses'][frame_idx]
            elif pose_type == 'icp':
                return self.precomputed_poses['icp_poses'][frame_idx]
            else:
                print(f"경고: 알 수 없는 pose 타입: {pose_type}")
                return None
        except (KeyError, IndexError) as e:
            print(f"경고: 프레임 {frame_idx}의 pose를 찾을 수 없습니다: {e}")
            return None

    def project_lane_to_3d(self, white_mask_roi, frame_idx):
        """차선 검출 결과를 라이다 포인트와 매핑하여 3D 좌표로 변환"""
        # 1. 라이다 포인트 로드
        if frame_idx >= len(self.velodyne_files):
            return None
            
        points = np.fromfile(self.velodyne_files[frame_idx], dtype=np.float32).reshape(-1, 4)
        points_xyz = points[:, :3]
        
        # 2. 라이다 포인트를 이미지 평면에 투영
        points_homo = np.ones((len(points_xyz), 4))
        points_homo[:, :3] = points_xyz
        
        # 라이다 -> 카메라 변환
        points_cam = (self.Tr_velo_to_cam @ points_homo.T).T
        
        # 정규화
        points_rect = (self.R0_rect @ points_cam[:, :3].T).T
        
        # 카메라 좌표를 이미지 평면에 투영
        points_proj = (self.P2 @ np.vstack((points_rect.T, np.ones(len(points_rect))))).T
        points_proj = points_proj[:, :2] / points_proj[:, 2:]
        
        # 이미지 내부에 있는 포인트만 선택
        mask = (points_proj[:, 0] >= 0) & (points_proj[:, 0] < self.image_width) & \
               (points_proj[:, 1] >= 0) & (points_proj[:, 1] < self.image_height) & \
               (points_cam[:, 2] > 0)  # 카메라 앞쪽에 있는 포인트만
        
        valid_points_proj = points_proj[mask]
        valid_points_xyz = points_xyz[mask]
        
        if len(valid_points_proj) == 0:
            return None
        
        # 3. 차선 마스크 영역에 해당하는 라이다 포인트 찾기
        lane_points_3d = []
        
        # 투영된 라이다 포인트들 중에서 차선 마스크 영역에 있는 것들 선택
        for i, (u, v) in enumerate(valid_points_proj):
            u_int, v_int = int(round(u)), int(round(v))
            
            # 차선 마스크 영역 확인 (주변 픽셀도 확인하여 더 관대하게)
            window_size = 3  # 3x3 윈도우
            found_lane = False
            
            for du in range(-window_size, window_size + 1):
                for dv in range(-window_size, window_size + 1):
                    check_u = u_int + du
                    check_v = v_int + dv
                    
                    if (0 <= check_u < self.image_width and 
                        0 <= check_v < self.image_height and
                        white_mask_roi[check_v, check_u] > 0):
                        found_lane = True
                        break
                        
                if found_lane:
                    break
            
            if found_lane:
                lane_points_3d.append(valid_points_xyz[i])
        
        if len(lane_points_3d) == 0:
            return None
        
        lane_points_3d = np.array(lane_points_3d)
        
        # 4. 라이다 좌표를 월드 좌표로 변환
        if self.precomputed_poses is not None:
            pose = self.get_pose_from_precomputed(frame_idx, pose_type='gt')
            if pose is not None:
                # 4x4 변환 행렬
                transform = np.eye(4)
                transform[:3, :3] = pose[:3, :3]
                transform[:3, 3] = pose[:3, 3]
                
                # 동차 좌표로 변환
                lane_homo = np.hstack([lane_points_3d, np.ones((len(lane_points_3d), 1))])
                
                # 월드 좌표로 변환
                world_points = (transform @ lane_homo.T).T
                world_3d = world_points[:, :3]
                
                # 필터링: 차량 주변 합리적인 거리 내의 점들만
                distances = np.linalg.norm(world_3d[:, :2] - pose[:2, 3], axis=1)
                height_diff = np.abs(world_3d[:, 2] - pose[2, 3])
                
                valid_mask = (distances < 50.0) & (height_diff < 3.0) & (distances > 1.0)
                
                if np.any(valid_mask):
                    return world_3d[valid_mask]
        
        return None

    def visualize_trajectory_with_lanes(self, pointcloud_path, trajectory_path):
        """궤적과 차선 검출 결과를 함께 시각화"""
        # 1. 궤적 로드
        trajectory_data = np.load(trajectory_path)
        gt_trajectory = trajectory_data['gt_trajectory']
        icp_trajectory = trajectory_data['icp_trajectory']
        
        # 궤적 데이터 크기 확인
        num_trajectory_points = len(gt_trajectory)
        print(f"궤적 데이터 크기: {num_trajectory_points}")
        
        # 2. 시각화
        plt.figure(figsize=(20, 8))
        
        # 2.1 궤적만 표시
        plt.subplot(121)
        plt.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], 'r-', linewidth=3, label='GT Trajectory', alpha=0.9)
        plt.plot(icp_trajectory[:, 0], icp_trajectory[:, 1], 'b-', linewidth=3, label='ICP Trajectory', alpha=0.9)
        plt.scatter(gt_trajectory[0, 0], gt_trajectory[0, 1], c='green', s=200, marker='o', label='Start', zorder=5)
        plt.scatter(gt_trajectory[-1, 0], gt_trajectory[-1, 1], c='red', s=200, marker='X', label='End', zorder=5)
        plt.title('Vehicle Trajectory', fontsize=16, fontweight='bold')
        plt.xlabel('X Position (m)', fontsize=14)
        plt.ylabel('Y Position (m)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.legend(fontsize=12)
        
        # 2.2 차선 검출 결과 오버레이
        plt.subplot(122)
        plt.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], 'r-', linewidth=3, label='GT Trajectory', alpha=0.9)
        
        # 차선 검출 결과를 오버레이
        # 10프레임마다 처리
        max_frames = min(num_trajectory_points, len(self.image_files))
        frame_indices = np.arange(0, max_frames, 10)  # 10프레임마다 처리
        print(f"🔍 차선 검출 시작: 0부터 {max_frames-1}까지 10프레임 간격으로 총 {len(frame_indices)}개 프레임 처리")
        
        all_lane_points = []
        successful_frames = 0
        
        for i, frame_idx in enumerate(frame_indices):
            if i % 5 == 0:  # 5프레임마다 진행상황 출력
                print(f"⏳ 진행상황: {i+1}/{len(frame_indices)} ({(i+1)/len(frame_indices)*100:.1f}%)")
            
            # 이미지 로드
            if frame_idx >= len(self.image_files):
                continue
                
            image_path = self.image_files[frame_idx]
            if not os.path.exists(image_path):
                continue
                
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 차선 검출
            white_mask, white_mask_roi, edges = self.detect_white_lanes(image)
            
            # 차선 검출 결과를 3D 공간에 투영
            lane_points_3d = self.project_lane_to_3d(white_mask_roi, frame_idx)
            if lane_points_3d is not None and len(lane_points_3d) > 0:
                all_lane_points.append(lane_points_3d)
                successful_frames += 1
                
                # 차선 포인트를 노란색 계열로 표시
                plt.scatter(lane_points_3d[:, 0], lane_points_3d[:, 1], 
                          c='yellow', s=3, alpha=0.7, 
                          label='Lane Detection' if successful_frames == 1 else "")
                
                if i % 10 == 0:  # 10프레임마다 상세 정보 출력
                    print(f"  ✅ 프레임 {frame_idx}: {len(lane_points_3d)}개 차선 포인트 검출됨")
        
        print(f"🎯 차선 검출 완료: {successful_frames}/{len(frame_indices)} 프레임에서 성공")
        
        # 시작점과 끝점 표시
        plt.scatter(gt_trajectory[0, 0], gt_trajectory[0, 1], c='green', s=200, marker='o', label='Start', zorder=5)
        plt.scatter(gt_trajectory[-1, 0], gt_trajectory[-1, 1], c='red', s=200, marker='X', label='End', zorder=5)
        
        plt.title('Trajectory with Lane Detection (BEV)', fontsize=16, fontweight='bold')
        plt.xlabel('X Position (m)', fontsize=14)
        plt.ylabel('Y Position (m)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.legend(fontsize=12)
        
        plt.tight_layout()
        
        # 결과 저장
        results_dir = Path('results_lane') / self.date
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f'kitti_trajectory_lanes_{timestamp}.png'
        filepath = results_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✅ 궤적과 차선 검출 결과가 저장되었습니다: {filepath}")
        
        # 통계 출력
        total_lane_points = sum(len(points) for points in all_lane_points)
        print(f"\n📊 차선 검출 통계:")
        print(f"  - 처리된 프레임 수: {len(all_lane_points)}")
        print(f"  - 총 차선 포인트 수: {total_lane_points}")
        print(f"  - 프레임당 평균 포인트 수: {total_lane_points/len(all_lane_points) if all_lane_points else 0:.1f}")

def main():
    parser = argparse.ArgumentParser(description='KITTI Lane Segmentation')
    parser.add_argument('--base_path', type=str, 
                      default='KITTI_dataset',
                      help='KITTI 데이터셋의 기본 경로')
    parser.add_argument('--dataset_prefix', type=str,
                      default='2011_09_26_drive_0001',
                      help='데이터셋 프리픽스 (예: 2011_09_26_drive_0001)')
    parser.add_argument('--output_dir', type=str,
                      default='results',
                      help='결과를 저장할 디렉토리')
    parser.add_argument('--pose_file', type=str,
                      help='사전 계산된 pose 파일 경로')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    results_dir = Path(args.output_dir) / args.dataset_prefix
    os.makedirs(results_dir, exist_ok=True)
    
    # pose 파일 경로가 지정되지 않은 경우 기본값 사용
    if args.pose_file is None:
        pose_files = sorted(glob.glob(str(Path(args.output_dir) / args.dataset_prefix / 'kitti_poses_*.pkl')))
        if pose_files:
            args.pose_file = pose_files[-1]  # 가장 최근 파일 사용
            print(f"자동으로 최근 pose 파일을 사용합니다: {args.pose_file}")
        else:
            print("경고: pose 파일을 찾을 수 없습니다.")
    
    # 차선 분할 모델 초기화
    lane_seg = KITTILaneSegmentation(args.base_path, args.dataset_prefix, args.pose_file)
    
    print("KITTI Lane Segmentation 시작...")
    print(f"데이터셋: {args.dataset_prefix}")
    
    # 여러 프레임에 대해 차선 분할 수행 (10프레임마다)
    max_frames = len(lane_seg.image_files)
    frame_indices = np.arange(0, max_frames, 10)
    print(f"처리할 프레임: 0부터 {max_frames-1}까지 10프레임 간격으로 총 {len(frame_indices)}개")
    
    for i, frame_idx in enumerate(frame_indices):
        print(f"\n프레임 {frame_idx} 처리 중... ({i+1}/{len(frame_indices)})")
        # 마지막 프레임에서만 라이다 포인트 표시
        show_lidar = (i == len(frame_indices) - 1)
        lane_seg.visualize_lane_segmentation(frame_idx, show_lidar=show_lidar)
    
    # 궤적과 차선 검출 결과 시각화
    pointcloud_path = results_dir / 'kitti_gt_pointcloud.pcd'
    trajectory_path = results_dir / 'kitti_trajectory.npz'
    lane_seg.visualize_trajectory_with_lanes(pointcloud_path, trajectory_path)
    
    print(f"\n완료! 결과가 {results_dir} 디렉토리에 저장되었습니다.")

if __name__ == "__main__":
    main() 