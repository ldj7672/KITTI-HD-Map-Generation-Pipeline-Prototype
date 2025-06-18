import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.spatial.transform import Rotation
import os
import glob
from mpl_toolkits.mplot3d import Axes3D
import cv2
import open3d as o3d
import pickle
import json
from datetime import datetime
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class KITTITrackletsVisualizer:
    def __init__(self, base_path, dataset_prefix):
        self.base_path = Path(base_path)
        self.date = dataset_prefix
        self.drive = f'{dataset_prefix}_sync'
        
        # 데이터 경로 설정
        self.data_path = self.base_path / 'data' / self.drive
        self.tracklets_path = self.base_path / 'tracklets' / self.drive
        self.image_path = self.data_path / 'image_02' / 'data'  # 2D 이미지 경로
        self.velodyne_path = self.data_path / 'velodyne_points' / 'data'  # LiDAR 경로 추가
        
        # XML 파일 경로
        self.tracklets_file = self.tracklets_path / 'tracklet_labels.xml'
        
        # 파일 존재 여부 확인
        if not self.tracklets_file.exists():
            raise FileNotFoundError(f"Tracklets 파일을 찾을 수 없습니다: {self.tracklets_file}")
        
        print(f"Tracklets 파일 로드: {self.tracklets_file}")
        
        # Velodyne 파일 목록
        self.velodyne_files = sorted(glob.glob(str(self.velodyne_path / '*.bin')))
        print(f"Velodyne 파일 수: {len(self.velodyne_files)}개")
        
        # 시각화 결과 저장
        self.visualization_results = []
        
        # Ego pose 데이터 (나중에 로드)
        self.ego_poses = None
        self.pose_data = None
        
        # GT 포인트클라우드와 trajectory (나중에 로드)
        self.gt_pointcloud = None
        self.trajectory = None
        
        # KITTI 캘리브레이션 행렬 (기본값)
        # P2: 카메라 투영 행렬
        self.P2 = np.array([
            [7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01],
            [0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-01],
            [0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03]
        ])
        
        # Tr_velo_to_cam: Velodyne to camera coordinate transformation
        self.Tr_velo_to_cam = np.array([
            [7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03],
            [1.480249e-02, 7.280733e-04, -9.998902e-01, -7.631618e-02],
            [9.998621e-01, 7.523790e-03, 1.480755e-02, -2.717806e-01],
            [0.0, 0.0, 0.0, 1.0]
        ])
        
        # R0_rect: 보정 회전 행렬
        self.R0_rect = np.array([
            [9.999239e-01, 9.837760e-03, -7.445048e-03, 0.0],
            [-9.869795e-03, 9.999421e-01, -4.278459e-03, 0.0],
            [7.402527e-03, 4.351614e-03, 9.999631e-01, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        
    def load_tracklets(self, frame_idx):
        """XML 파일에서 특정 프레임의 tracklets 로드"""
        tracklets = []
        tree = ET.parse(self.tracklets_file)
        root = tree.getroot()
        
        print(f"\nXML 파일 파싱 시작 (프레임 {frame_idx})...")
        
        # tracklets 요소 찾기
        tracklets_elem = root.find('tracklets')
        if tracklets_elem is None:
            print("tracklets 요소를 찾을 수 없습니다.")
            return tracklets
            
        # tracklets 아래의 item 요소들 찾기
        tracklet_items = tracklets_elem.findall('item')
        print(f"발견된 총 tracklet 수: {len(tracklet_items)}")
        
        for tracklet_idx, tracklet in enumerate(tracklet_items):
            try:
                # 객체 정보 추출
                object_type = tracklet.find('objectType').text
                h = float(tracklet.find('h').text)
                w = float(tracklet.find('w').text)
                l = float(tracklet.find('l').text)
                
                # first_frame 정보
                first_frame = int(tracklet.find('first_frame').text)
                
                # poses 요소 찾기
                poses = tracklet.find('poses')
                if poses is None:
                    continue
                
                # 해당 프레임의 pose 찾기
                pose_items = poses.findall('item')
                for pose_idx, pose in enumerate(pose_items):
                    current_frame = first_frame + pose_idx
                    
                    if current_frame == frame_idx:
                        # 위치 및 회전 정보
                        tx = float(pose.find('tx').text)
                        ty = float(pose.find('ty').text)
                        tz = float(pose.find('tz').text)
                        rx = float(pose.find('rx').text)
                        ry = float(pose.find('ry').text)
                        rz = float(pose.find('rz').text)
                        
                        tracklet_data = {
                            'type': object_type,
                            'dimensions': [h, w, l],
                            'location': [tx, ty, tz],
                            'rotation': [rx, ry, rz]
                        }
                        tracklets.append(tracklet_data)
                        print(f"프레임 {frame_idx}에 {object_type} 객체 추가됨 (tracklet {tracklet_idx})")
            
            except Exception as e:
                print(f"tracklet {tracklet_idx} 처리 중 오류 발생: {str(e)}")
                continue
        
        print(f"프레임 {frame_idx}의 총 tracklet 수: {len(tracklets)}")
        return tracklets
        
    def velo_to_cam(self, points):
        """Velodyne 좌표계를 카메라 좌표계로 변환"""
        points_hom = np.hstack([points, np.ones((points.shape[0], 1))])
        points_cam = np.dot(self.Tr_velo_to_cam, points_hom.T).T
        return points_cam[:, :3]
    
    def cam_to_rect(self, points):
        """카메라 좌표계를 보정된 좌표계로 변환"""
        points_hom = np.hstack([points, np.ones((points.shape[0], 1))])
        points_rect = np.dot(self.R0_rect, points_hom.T).T
        return points_rect[:, :3]
    
    def rect_to_img(self, points):
        """보정된 3D 좌표를 2D 이미지 좌표로 투영"""
        points_hom = np.hstack([points, np.ones((points.shape[0], 1))])
        points_2d_hom = np.dot(self.P2, points_hom.T).T
        points_2d = points_2d_hom[:, :2] / points_2d_hom[:, 2:3]
        return points_2d
    
    def get_3d_box_corners(self, dimensions, location, rotation):
        """3D 박스의 8개 꼭지점 계산"""
        h, w, l = dimensions
        x, y, z = location
        
        # KITTI tracklet에서 z는 바닥면 기준이므로, 박스를 바닥면에서 위로 h만큼 생성
        corners = np.array([
            [-l/2, -w/2, 0],  [l/2, -w/2, 0],  [l/2, w/2, 0],  [-l/2, w/2, 0],     # 바닥면 (z=0)
            [-l/2, -w/2, h],  [l/2, -w/2, h],  [l/2, w/2, h],  [-l/2, w/2, h]      # 윗면 (z=h)
        ])
        
        # Z축 회전 적용 (rz)
        rz = rotation[2]
        rot_matrix = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1]
        ])
        corners = np.dot(corners, rot_matrix.T)
        
        # 위치 이동 (바닥면 기준)
        corners += np.array([x, y, z])
        
        return corners
    
    def visualize_tracklets(self, tracklets, frame_idx):
        """2D 이미지 위에 3D 박스 시각화"""
        # 이미지 로드
        image_file = self.image_path / f"{frame_idx:010d}.png"
        if not image_file.exists():
            print(f"이미지 파일을 찾을 수 없습니다: {image_file}")
            return None
            
        image = cv2.imread(str(image_file))
        if image is None:
            print(f"이미지를 읽을 수 없습니다: {image_file}")
            return None
            
        # 색상 매핑
        colors = {
            'Car': (0, 0, 255),      # 빨강
            'Van': (255, 0, 0),      # 파랑
            'Truck': (0, 255, 0),    # 초록
            'Pedestrian': (0, 255, 255),  # 노랑
            'Person_sitting': (255, 0, 255),  # 보라
            'Cyclist': (0, 165, 255),  # 주황
            'Tram': (128, 0, 128),   # 자주
            'Misc': (128, 128, 128)  # 회색
        }
        
        for tracklet in tracklets:
            color = colors.get(tracklet['type'], (128, 128, 128))
            
            try:
                print(f"처리 중: {tracklet['type']}")
                print(f"  크기(h,w,l): {tracklet['dimensions']}")
                print(f"  위치(x,y,z): {tracklet['location']}")
                print(f"  회전(rx,ry,rz): {tracklet['rotation']}")
                
                # 3D 박스의 8개 꼭지점 계산
                corners_3d = self.get_3d_box_corners(
                    tracklet['dimensions'], 
                    tracklet['location'], 
                    tracklet['rotation']
                )
                
                print(f"  3D 꼭지점 범위: Z축 {corners_3d[:, 2].min():.2f} ~ {corners_3d[:, 2].max():.2f}")
                
                # 좌표계 변환: Velodyne -> Camera -> Rectified
                corners_cam = self.velo_to_cam(corners_3d)
                corners_rect = self.cam_to_rect(corners_cam)
                
                print(f"  변환 후 Z축 범위: {corners_rect[:, 2].min():.2f} ~ {corners_rect[:, 2].max():.2f}")
                
                # 카메라 앞에 있는 점들만 필터링
                mask = corners_rect[:, 2] > 0.1
                if not np.any(mask):
                    print(f"  모든 점이 카메라 뒤에 있음 - 스킵")
                    continue
                
                # 2D 투영
                corners_2d = self.rect_to_img(corners_rect)
                
                print(f"  2D 투영 범위: X {corners_2d[:, 0].min():.0f}~{corners_2d[:, 0].max():.0f}, Y {corners_2d[:, 1].min():.0f}~{corners_2d[:, 1].max():.0f}")
                
                # 이미지 경계 내에 있는지 확인
                h, w = image.shape[:2]
                valid_points = []
                for i, (x, y) in enumerate(corners_2d):
                    if 0 <= x <= w and 0 <= y <= h and mask[i]:
                        valid_points.append((int(x), int(y)))
                
                print(f"  유효한 점 개수: {len(valid_points)}/8")
                
                if len(valid_points) < 4:  # 최소 4개 점이 필요
                    print(f"  유효한 점이 부족함 - 스킵")
                    continue
                
                # 3D 박스 그리기 - 면 단위로 처리
                self.draw_3d_box(image, corners_2d, mask, color)
                
                print(f"  3D 박스 그리기 완료!")
                
                # 객체 타입과 크기 정보 표시
                if valid_points:
                    text_pos = valid_points[0]
                    h_dim, w_dim, l_dim = tracklet['dimensions']
                    x, y, z = tracklet['location']
                    text = f"{tracklet['type']} H:{h_dim:.1f} Z:{z:.1f}"
                    cv2.putText(image, text,
                              (text_pos[0], text_pos[1] - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # 객체 바닥면 중심점을 작은 원으로 표시
                    center_3d = np.array([[x, y, z]])  # 바닥면 중심
                    center_cam = self.velo_to_cam(center_3d)
                    center_rect = self.cam_to_rect(center_cam)
                    if center_rect[0, 2] > 0.1:
                        center_2d = self.rect_to_img(center_rect)
                        center_x, center_y = int(center_2d[0, 0]), int(center_2d[0, 1])
                        h_img, w_img = image.shape[:2]
                        if 0 <= center_x <= w_img and 0 <= center_y <= h_img:
                            cv2.circle(image, (center_x, center_y), 5, (255, 255, 255), -1)
                            cv2.circle(image, (center_x, center_y), 5, color, 2)
                    
                    # 객체 실제 중심점도 표시 (높이의 중간)
                    real_center_3d = np.array([[x, y, z + h_dim/2]])  # 실제 중심
                    real_center_cam = self.velo_to_cam(real_center_3d)
                    real_center_rect = self.cam_to_rect(real_center_cam)
                    if real_center_rect[0, 2] > 0.1:
                        real_center_2d = self.rect_to_img(real_center_rect)
                        real_center_x, real_center_y = int(real_center_2d[0, 0]), int(real_center_2d[0, 1])
                        h_img, w_img = image.shape[:2]
                        if 0 <= real_center_x <= w_img and 0 <= real_center_y <= h_img:
                            cv2.circle(image, (real_center_x, real_center_y), 3, (0, 255, 255), -1)  # 노란색 점
                
                print(f"  시각화 완료!")
                
            except Exception as e:
                print(f"객체 시각화 중 오류: {e}")
                continue
        
        # BEV 이미지 생성
        points = None
        if frame_idx < len(self.velodyne_files):
            points = self.load_velodyne_points(self.velodyne_files[frame_idx])
        
        bev_image = self.create_bev_visualization(tracklets, frame_idx, points)
        
        # 2D 이미지와 BEV를 위아래로 결합
        combined_image = self.combine_2d_and_bev(image, bev_image, frame_idx)
        
        # 결과 저장
        results_dir = Path('results_tracklets') / self.date
        os.makedirs(results_dir, exist_ok=True)
        
        filename = f'combined_frame{frame_idx:06d}.png'
        filepath = results_dir / filename
        
        print(f"결합된 이미지 저장: {filepath}")
        cv2.imwrite(str(filepath), combined_image)
        print(f"이미지 저장 완료!")
        
        return filepath
    
    def combine_2d_and_bev(self, image_2d, bev_image, frame_idx):
        """2D 이미지와 BEV 이미지를 위아래로 결합"""
        if bev_image is None:
            # BEV가 없으면 2D 이미지만 반환
            return image_2d
        
        # 이미지 크기 정보
        img_h, img_w = image_2d.shape[:2]
        bev_h, bev_w = bev_image.shape[:2]
        
        # BEV를 2D 이미지 너비에 맞춰 리사이즈
        if bev_w != img_w:
            aspect_ratio = bev_h / bev_w
            new_bev_h = int(img_w * aspect_ratio)
            bev_image_resized = cv2.resize(bev_image, (img_w, new_bev_h))
        else:
            bev_image_resized = bev_image
            new_bev_h = bev_h
        
        # 결합된 이미지 생성 (위: 2D, 아래: BEV)
        total_height = img_h + new_bev_h + 20  # 20픽셀 간격
        combined_image = np.zeros((total_height, img_w, 3), dtype=np.uint8)
        
        # 2D 이미지를 위쪽에 배치
        combined_image[:img_h, :] = image_2d
        
        # BEV 이미지를 아래쪽에 배치 (간격 포함)
        combined_image[img_h + 20:img_h + 20 + new_bev_h, :] = bev_image_resized
        
        # 구분선 그리기
        cv2.line(combined_image, (0, img_h + 10), (img_w, img_h + 10), (255, 255, 255), 2)
        
        # 텍스트 라벨 추가
        cv2.putText(combined_image, f'2D Image View - Frame {frame_idx}', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(combined_image, 'Bird\'s Eye View (BEV)', 
                   (10, img_h + 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        return combined_image
    
    def draw_3d_box(self, image, corners_2d, mask, color):
        """3D 박스를 면 단위로 그리기 (더 나은 시각화)"""
        h, w = image.shape[:2]
        
        # 8개 꼭지점의 인덱스
        # 바닥면: 0,1,2,3 / 윗면: 4,5,6,7
        
        # 박스의 6개 면 정의 (시계방향)
        faces = [
            [0, 1, 2, 3],  # 바닥면
            [4, 7, 6, 5],  # 윗면
            [0, 4, 5, 1],  # 앞면
            [2, 6, 7, 3],  # 뒷면
            [0, 3, 7, 4],  # 왼쪽면
            [1, 5, 6, 2]   # 오른쪽면
        ]
        
        # 각 면에 대해 처리
        for face_idx, face in enumerate(faces):
            # 이 면의 모든 점이 카메라 앞에 있는지 확인
            face_mask = [mask[i] for i in face]
            if not all(face_mask):
                continue
            
            # 면의 꼭지점들을 가져오기
            face_points = []
            valid_face = True
            
            for i in face:
                x, y = corners_2d[i]
                # 이미지 경계를 넘어가는 점들을 클리핑
                x = max(0, min(w-1, int(x)))
                y = max(0, min(h-1, int(y)))
                face_points.append([x, y])
            
            if len(face_points) >= 3:  # 최소 3개 점이 있어야 면을 그릴 수 있음
                face_points = np.array(face_points, dtype=np.int32)
                
                # 면의 깊이 계산 (Z값의 평균)
                face_depths = [corners_2d[i] for i in face if mask[i]]
                if len(face_depths) > 0:
                    # 반투명 면 그리기
                    overlay = image.copy()
                    
                    # 면 색상 (약간 투명하게)
                    if face_idx == 0:  # 바닥면
                        face_color = tuple(int(c * 0.3) for c in color)
                    elif face_idx == 1:  # 윗면
                        face_color = tuple(int(c * 0.5) for c in color)
                    else:  # 측면들
                        face_color = tuple(int(c * 0.4) for c in color)
                    
                    cv2.fillPoly(overlay, [face_points], face_color)
                    cv2.addWeighted(image, 0.7, overlay, 0.3, 0, image)
                    
                    # 면의 경계선 그리기
                    cv2.polylines(image, [face_points], True, color, 2)
        
        # 주요 모서리를 강조하여 그리기
        key_edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # 바닥면
            [4, 5], [5, 6], [6, 7], [7, 4],  # 윗면
            [0, 4], [1, 5], [2, 6], [3, 7]   # 수직 모서리
        ]
        
        for edge in key_edges:
            p1_idx, p2_idx = edge
            if mask[p1_idx] and mask[p2_idx]:
                p1 = corners_2d[p1_idx]
                p2 = corners_2d[p2_idx]
                
                # 라인 클리핑
                x1, y1 = max(0, min(w-1, int(p1[0]))), max(0, min(h-1, int(p1[1])))
                x2, y2 = max(0, min(w-1, int(p2[0]))), max(0, min(h-1, int(p2[1])))
                
                cv2.line(image, (x1, y1), (x2, y2), color, 3)
    
    def load_ego_poses(self, pose_file_path):
        """Ego pose 데이터 로드 (pickle 또는 json)"""
        pose_path = Path(pose_file_path)
        
        if not pose_path.exists():
            print(f"Pose 파일을 찾을 수 없습니다: {pose_path}")
            return False
            
        try:
            if pose_path.suffix == '.pkl':
                with open(pose_path, 'rb') as f:
                    self.pose_data = pickle.load(f)
                self.ego_poses = self.pose_data.get('fused_poses', self.pose_data.get('gt_poses', []))
            elif pose_path.suffix == '.json':
                with open(pose_path, 'r') as f:
                    self.pose_data = json.load(f)
                # JSON에서 pose 행렬 재구성 필요
                self.ego_poses = []
                positions = self.pose_data.get('estimated_positions', [])
                rotations = self.pose_data.get('estimated_rotations', [])
                
                for pos, rot in zip(positions, rotations):
                    pose_matrix = np.eye(4)
                    pose_matrix[:3, 3] = pos
                    rotation_matrix = Rotation.from_euler('xyz', rot).as_matrix()
                    pose_matrix[:3, :3] = rotation_matrix
                    self.ego_poses.append(pose_matrix)
            
            print(f"Ego pose 데이터 로드 완료: {len(self.ego_poses)}개 pose")
            return True
            
        except Exception as e:
            print(f"Pose 파일 로드 중 오류: {e}")
            return False
    
    def load_velodyne_points(self, file_path):
        """라이다 포인트 클라우드 로드"""
        points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
        return points[:, :3]  # x, y, z만 사용
    
    def create_bev_visualization(self, tracklets, frame_idx, points=None):
        """BEV(Bird's Eye View) 시각화 생성"""
        # BEV 설정 (X축: 진행방향 ±50m, Y축: 좌우방향 ±25m - 50% 줄임)
        bev_range_x = 50  # 진행방향 범위 (앞뒤)
        bev_range_y = 25  # 좌우방향 범위 (25%씩 줄임)
        bev_resolution = 0.1  # 0.1m per pixel
        bev_width = int(2 * bev_range_x / bev_resolution)   # X축 기준 너비 (진행방향)
        bev_height = int(2 * bev_range_y / bev_resolution)  # Y축 기준 높이 (좌우방향)
        
        # BEV 이미지 생성
        bev_image = np.zeros((bev_height, bev_width, 3), dtype=np.uint8)
        
        # 포인트 클라우드가 있으면 배경으로 표시
        if points is not None:
            # 범위 내 포인트만 필터링
            mask = (np.abs(points[:, 0]) < bev_range_x) & (np.abs(points[:, 1]) < bev_range_y)
            filtered_points = points[mask]
            
            if len(filtered_points) > 0:
                # 포인트를 픽셀 좌표로 변환
                # X축(진행방향): 왼쪽이 뒤쪽(-X), 오른쪽이 앞쪽(+X)
                # Y축(좌우방향): 위쪽이 왼쪽(+Y), 아래쪽이 오른쪽(-Y) - 반전 수정
                pixel_x = ((filtered_points[:, 0] + bev_range_x) / bev_resolution).astype(int)    # X 좌표
                pixel_y = ((bev_range_y - filtered_points[:, 1]) / bev_resolution).astype(int)    # Y 좌표 반전
                
                # 유효한 픽셀만 선택
                valid_mask = (pixel_x >= 0) & (pixel_x < bev_width) & (pixel_y >= 0) & (pixel_y < bev_height)
                pixel_x = pixel_x[valid_mask]
                pixel_y = pixel_y[valid_mask]
                
                # 포인트 클라우드를 회색으로 표시
                bev_image[pixel_y, pixel_x] = [50, 50, 50]
        
        # 색상 매핑
        colors = {
            'Car': (0, 0, 255),      # 빨강
            'Van': (255, 0, 0),      # 파랑
            'Truck': (0, 255, 0),    # 초록
            'Pedestrian': (0, 255, 255),  # 노랑
            'Person_sitting': (255, 0, 255),  # 보라
            'Cyclist': (0, 165, 255),  # 주황
            'Tram': (128, 0, 128),   # 자주
            'Misc': (128, 128, 128)  # 회색
        }
        
        for tracklet in tracklets:
            color = colors.get(tracklet['type'], (128, 128, 128))
            
            # 3D 박스의 바닥면 꼭지점만 계산 (BEV용)
            h, w, l = tracklet['dimensions']
            x, y, z = tracklet['location']
            rz = tracklet['rotation'][2]  # Z축 회전만 사용
            
            # 바닥면 4개 꼭지점
            corners = np.array([
                [-l/2, -w/2], [l/2, -w/2], [l/2, w/2], [-l/2, w/2]
            ])
            
            # 회전 적용
            rot_matrix = np.array([
                [np.cos(rz), -np.sin(rz)],
                [np.sin(rz), np.cos(rz)]
            ])
            corners = np.dot(corners, rot_matrix.T)
            
            # 위치 이동
            corners += np.array([x, y])
            
            # 픽셀 좌표로 변환
            pixel_corners = []
            for corner in corners:
                px = int((corner[0] + bev_range_x) / bev_resolution)    # X 좌표
                py = int((bev_range_y - corner[1]) / bev_resolution)    # Y 좌표 반전
                if 0 <= px < bev_width and 0 <= py < bev_height:
                    pixel_corners.append([px, py])  # OpenCV는 (x, y) 순서
            
            if len(pixel_corners) >= 3:
                # 박스 채우기
                pixel_corners = np.array(pixel_corners, dtype=np.int32)
                cv2.fillPoly(bev_image, [pixel_corners], color)
                cv2.polylines(bev_image, [pixel_corners], True, (255, 255, 255), 2)
                
                # 객체 타입 표시
                center_px = int((x + bev_range_x) / bev_resolution)
                center_py = int((bev_range_y - y) / bev_resolution)
                if 0 <= center_px < bev_width and 0 <= center_py < bev_height:
                    cv2.putText(bev_image, tracklet['type'][:3],
                              (center_px-15, center_py+5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # 차량 위치 및 방향 표시 (원점 = 차량 위치)
        vehicle_center_x = int(bev_range_x / bev_resolution)  # 차량 위치 (이미지 왼쪽)
        vehicle_center_y = int(bev_range_y / bev_resolution)  # 차량 위치 (이미지 가운데)
        
        # 차량 표시 (작은 사각형)
        vehicle_size = 15
        cv2.rectangle(bev_image, 
                     (vehicle_center_x - vehicle_size//2, vehicle_center_y - vehicle_size//2),
                     (vehicle_center_x + vehicle_size//2, vehicle_center_y + vehicle_size//2),
                     (0, 255, 255), -1)  # 노란색 차량
        
        # 차량 방향 화살표 (오른쪽 방향으로 진행)
        arrow_length = 30
        arrow_end_x = vehicle_center_x + arrow_length  # 오른쪽으로 화살표
        arrow_end_y = vehicle_center_y
        cv2.arrowedLine(bev_image, 
                       (vehicle_center_x, vehicle_center_y),
                       (arrow_end_x, arrow_end_y),
                       (0, 255, 255), 3, tipLength=0.3)  # 노란색 화살표
        
        # 좌표축 표시
        axis_length = 40
        # X축 (진행방향) - 수평선
        cv2.line(bev_image, 
                (vehicle_center_x - axis_length, vehicle_center_y + 60), 
                (vehicle_center_x + axis_length, vehicle_center_y + 60), 
                (255, 255, 255), 2)
        # Y축 (좌우방향) - 수직선
        cv2.line(bev_image, 
                (vehicle_center_x + 50, vehicle_center_y - axis_length), 
                (vehicle_center_x + 50, vehicle_center_y + axis_length), 
                (255, 255, 255), 2)
        
        # 축 라벨
        cv2.putText(bev_image, 'X(Forward)', 
                   (vehicle_center_x + axis_length + 5, vehicle_center_y + 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(bev_image, 'Y(Left)', 
                   (vehicle_center_x + 55, vehicle_center_y - axis_length - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 범위 정보 표시
        cv2.putText(bev_image, f'Range: X=±{bev_range_x}m, Y=±{bev_range_y}m', 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(bev_image, f'Resolution: {bev_resolution}m/pixel', 
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        return bev_image
    
    def create_multi_frame_bev(self, start_frame=0, num_frames=10, frame_step=1):
        """다중 프레임 BEV 시각화 (ego pose 기반 포인트 클라우드 정합)"""
        if self.ego_poses is None:
            print("Ego pose 데이터가 없습니다. load_ego_poses()를 먼저 호출하세요.")
            return None
        
        # BEV 설정 (X축: 진행방향 ±150m, Y축: 좌우방향 ±100m - GT 전체 영역 커버)
        bev_range_x = 150  # 진행방향 범위 (GT 전체 영역용)
        bev_range_y = 100  # 좌우방향 범위 (GT 전체 영역용)
        bev_resolution = 0.3  # 0.3m per pixel (해상도 조정)
        bev_width = int(2 * bev_range_x / bev_resolution)   # X축 기준 너비 (진행방향)
        bev_height = int(2 * bev_range_y / bev_resolution)  # Y축 기준 높이 (좌우방향)
        
        # BEV 이미지 생성
        bev_image = np.zeros((bev_height, bev_width, 3), dtype=np.uint8)
        
        # GT 포인트클라우드 표시 (배경으로)
        if self.gt_pointcloud is not None:
            print("GT 포인트클라우드를 BEV에 표시 중...")
            
            # 범위 내 포인트만 필터링
            mask = (np.abs(self.gt_pointcloud[:, 0]) < bev_range_x) & (np.abs(self.gt_pointcloud[:, 1]) < bev_range_y)
            filtered_gt_points = self.gt_pointcloud[mask]
            
            if len(filtered_gt_points) > 0:
                # 포인트를 픽셀 좌표로 변환
                pixel_x = ((filtered_gt_points[:, 0] + bev_range_x) / bev_resolution).astype(int)
                pixel_y = ((bev_range_y - filtered_gt_points[:, 1]) / bev_resolution).astype(int)
                
                # 유효한 픽셀만 선택
                valid_mask = (pixel_x >= 0) & (pixel_x < bev_width) & (pixel_y >= 0) & (pixel_y < bev_height)
                pixel_x = pixel_x[valid_mask]
                pixel_y = pixel_y[valid_mask]
                
                # GT 포인트클라우드를 어두운 회색으로 표시
                bev_image[pixel_y, pixel_x] = [40, 40, 40]
                print(f"GT 포인트클라우드 {len(filtered_gt_points)}개 포인트 표시 완료")
        
        # Trajectory 표시
        if self.trajectory is not None:
            print("Trajectory를 BEV에 표시 중...")
            
            # trajectory가 2D인지 3D인지 확인
            if self.trajectory.shape[1] >= 2:
                traj_points = self.trajectory[:, :2]  # X, Y만 사용
                
                # 범위 내 trajectory 포인트만 필터링
                mask = (np.abs(traj_points[:, 0]) < bev_range_x) & (np.abs(traj_points[:, 1]) < bev_range_y)
                filtered_traj = traj_points[mask]
                
                if len(filtered_traj) > 1:
                    # trajectory를 라인으로 그리기
                    prev_pixel = None
                    for point in filtered_traj:
                        pixel_x = int((point[0] + bev_range_x) / bev_resolution)
                        pixel_y = int((bev_range_y - point[1]) / bev_resolution)
                        
                        if 0 <= pixel_x < bev_width and 0 <= pixel_y < bev_height:
                            current_pixel = (pixel_x, pixel_y)
                            
                            if prev_pixel is not None:
                                cv2.line(bev_image, prev_pixel, current_pixel, (0, 255, 0), 3)  # 초록색 trajectory
                            
                            # trajectory 포인트 표시
                            cv2.circle(bev_image, current_pixel, 2, (0, 255, 0), -1)
                            prev_pixel = current_pixel
                    
                    print(f"Trajectory {len(filtered_traj)}개 포인트 표시 완료")
        
        # 다중 프레임 tracklet 수집
        all_tracklets = []
        
        end_frame = min(start_frame + num_frames * frame_step, len(self.ego_poses), len(self.velodyne_files))
        
        print(f"다중 프레임 tracklet 수집: 프레임 {start_frame} ~ {end_frame-1}")
        
        for frame_idx in range(start_frame, end_frame, frame_step):
            # tracklets 로드
            tracklets = self.load_tracklets(frame_idx)
            
            if tracklets and frame_idx < len(self.ego_poses):
                # Ego pose로 tracklet 위치를 월드 좌표계로 변환
                ego_pose = self.ego_poses[frame_idx]
                
                for tracklet in tracklets:
                    # tracklet 위치를 homogeneous 좌표로 변환
                    local_pos = np.array([tracklet['location'][0], tracklet['location'][1], tracklet['location'][2], 1])
                    
                    # 월드 좌표계로 변환
                    world_pos = ego_pose @ local_pos
                    
                    # 변환된 tracklet 생성
                    world_tracklet = tracklet.copy()
                    world_tracklet['location'] = world_pos[:3].tolist()
                    world_tracklet['frame_idx'] = frame_idx
                    all_tracklets.append(world_tracklet)
        
        # 색상 매핑 (프레임별로 조금씩 다르게)
        base_colors = {
            'Car': [0, 0, 255],      # 빨강 기반
            'Van': [255, 0, 0],      # 파랑 기반
            'Truck': [0, 255, 0],    # 초록 기반
            'Pedestrian': [0, 255, 255],  # 노랑 기반
            'Person_sitting': [255, 0, 255],  # 보라 기반
            'Cyclist': [0, 165, 255],  # 주황 기반
            'Tram': [128, 0, 128],   # 자주 기반
            'Misc': [128, 128, 128]  # 회색 기반
        }
        
        # tracklets를 프레임별로 그룹화
        frame_tracklets = {}
        for tracklet in all_tracklets:
            frame_idx = tracklet['frame_idx']
            if frame_idx not in frame_tracklets:
                frame_tracklets[frame_idx] = []
            frame_tracklets[frame_idx].append(tracklet)
        
        # 각 프레임의 tracklets를 BEV에 그리기
        for frame_idx, tracklets in frame_tracklets.items():
            # 프레임에 따른 투명도 계산 (최신 프레임일수록 진하게)
            alpha = 0.4 + 0.6 * (frame_idx - start_frame) / max(1, num_frames * frame_step - 1)
            
            for tracklet in tracklets:
                base_color = base_colors.get(tracklet['type'], [128, 128, 128])
                color = tuple(int(c * alpha) for c in base_color)
                
                # 3D 박스의 바닥면 꼭지점 계산
                h, w, l = tracklet['dimensions']
                x, y, z = tracklet['location']
                rz = tracklet['rotation'][2]
                
                # 바닥면 4개 꼭지점
                corners = np.array([
                    [-l/2, -w/2], [l/2, -w/2], [l/2, w/2], [-l/2, w/2]
                ])
                
                # 회전 적용
                rot_matrix = np.array([
                    [np.cos(rz), -np.sin(rz)],
                    [np.sin(rz), np.cos(rz)]
                ])
                corners = np.dot(corners, rot_matrix.T)
                
                # 위치 이동
                corners += np.array([x, y])
                
                # 픽셀 좌표로 변환
                pixel_corners = []
                for corner in corners:
                    px = int((corner[0] + bev_range_x) / bev_resolution)
                    py = int((bev_range_y - corner[1]) / bev_resolution)
                    if 0 <= px < bev_width and 0 <= py < bev_height:
                        pixel_corners.append([px, py])
                
                if len(pixel_corners) >= 3:
                    # 박스 채우기 (투명도 적용)
                    pixel_corners = np.array(pixel_corners, dtype=np.int32)
                    overlay = bev_image.copy()
                    cv2.fillPoly(overlay, [pixel_corners], color)
                    cv2.addWeighted(bev_image, 1-alpha*0.3, overlay, alpha*0.3, 0, bev_image)
                    cv2.polylines(bev_image, [pixel_corners], True, color, 2)
        
        # 범례와 정보 표시
        cv2.putText(bev_image, f'GT BEV with Tracklets (Frames {start_frame}-{end_frame-1})', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(bev_image, f'Range: X=±{bev_range_x}m, Y=±{bev_range_y}m', 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        cv2.putText(bev_image, f'GT Points: {len(self.gt_pointcloud) if self.gt_pointcloud is not None else 0}', 
                   (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        cv2.putText(bev_image, f'Trajectory Points: {len(self.trajectory) if self.trajectory is not None else 0}', 
                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        
        # 범례 색상 표시
        legend_y = bev_height - 150
        cv2.putText(bev_image, 'Legend:', (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(bev_image, '- Gray: GT PointCloud', (10, legend_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(bev_image, '- Green: GT Trajectory', (10, legend_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(bev_image, '- Colored: Tracklets', (10, legend_y + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return bev_image

    def process_sequence(self, start_idx=0, end_idx=None, enable_bev=True, enable_multi_frame_bev=True):
        """시퀀스 전체 처리 (2D + BEV 결합 시각화)"""
        if end_idx is None:
            end_idx = 100  # 기본값 설정
        
        for frame_idx in range(start_idx, end_idx):
            # tracklets 로드
            tracklets = self.load_tracklets(frame_idx)
            
            if tracklets:  # tracklets가 있는 경우에만 시각화
                print(f"\n프레임 {frame_idx} 처리 중...")
                print(f"검출된 객체 수: {len(tracklets)}")
                
                # 2D + BEV 결합 시각화
                vis_file = self.visualize_tracklets(tracklets, frame_idx)
                if vis_file:
                    self.visualization_results.append({
                        'frame_idx': frame_idx,
                        'tracklets_count': len(tracklets),
                        'visualization_file': str(vis_file)
                    })
                
                print(f"시각화 완료: {vis_file}")
                print("-" * 50)
            else:
                if frame_idx % 10 == 0:  # 10프레임마다 출력
                    print(f"프레임 {frame_idx}: 검출된 객체 없음")
        
        # GT 기반 전체 프레임 BEV 생성 (ego pose와 GT 데이터가 있을 때만)
        if enable_multi_frame_bev and self.ego_poses is not None:
            print("\nGT 기반 전체 프레임 BEV 생성 중...")
            
            # 전체 프레임을 하나로 합친 BEV 생성
            print(f"전체 프레임 통합 GT BEV 생성 중 (프레임 {start_idx}-{end_idx-1})...")
            
            all_frames_bev = self.create_multi_frame_bev(
                start_frame=start_idx, 
                num_frames=end_idx - start_idx, 
                frame_step=1
            )
            
            if all_frames_bev is not None:
                results_dir = Path('results_tracklets') / self.date
                os.makedirs(results_dir, exist_ok=True)
                gt_bev_filename = f'gt_bev_with_tracklets_{start_idx:06d}_{end_idx-1:06d}.png'
                gt_bev_filepath = results_dir / gt_bev_filename
                cv2.imwrite(str(gt_bev_filepath), all_frames_bev)
                print(f"GT 기반 전체 프레임 BEV 저장: {gt_bev_filepath}")
            else:
                print("GT BEV 생성 실패")
        else:
            if not enable_multi_frame_bev:
                print("GT BEV 생성이 비활성화되었습니다.")
            elif self.ego_poses is None:
                print("Ego pose 데이터가 없어 GT BEV를 생성할 수 없습니다.")

    def save_results(self, output_dir):
        """시각화 결과 저장"""
        results_dir = Path(output_dir) / self.date
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'tracklets_visualization_results_{timestamp}.json'
        filepath = results_dir / filename
        
        results_data = {
            'metadata': {
                'dataset_prefix': self.date,
                'total_frames': len(self.visualization_results),
                'timestamp': datetime.now().isoformat()
            },
            'results': self.visualization_results
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\n시각화 결과가 저장되었습니다: {filepath}")
        return filepath

    def load_gt_data(self, dataset_prefix):
        """GT 포인트클라우드와 trajectory 데이터 로드"""
        results_dir = Path('results') / dataset_prefix
        
        if not results_dir.exists():
            print(f"Results 디렉토리를 찾을 수 없습니다: {results_dir}")
            return False
        
        # GT 포인트클라우드 로드
        gt_pcd_file = results_dir / 'kitti_gt_pointcloud.pcd'
        if gt_pcd_file.exists():
            try:
                pcd = o3d.io.read_point_cloud(str(gt_pcd_file))
                self.gt_pointcloud = np.asarray(pcd.points)
                print(f"GT 포인트클라우드 로드 완료: {len(self.gt_pointcloud)}개 포인트")
            except Exception as e:
                print(f"GT 포인트클라우드 로드 실패: {e}")
                return False
        else:
            print(f"GT 포인트클라우드 파일을 찾을 수 없습니다: {gt_pcd_file}")
            return False
        
        # Trajectory 로드
        trajectory_file = results_dir / 'kitti_trajectory.npz'
        if trajectory_file.exists():
            try:
                traj_data = np.load(trajectory_file)
                # GT trajectory 추출
                if 'gt_trajectory' in traj_data:
                    self.trajectory = traj_data['gt_trajectory']
                elif 'trajectory' in traj_data:
                    self.trajectory = traj_data['trajectory']
                else:
                    # 키가 없는 경우 첫 번째 배열 사용
                    keys = list(traj_data.keys())
                    if keys:
                        self.trajectory = traj_data[keys[0]]
                
                if self.trajectory is not None:
                    print(f"Trajectory 로드 완료: {len(self.trajectory)}개 포인트")
                else:
                    print("Trajectory 데이터를 찾을 수 없습니다.")
                    
            except Exception as e:
                print(f"Trajectory 로드 실패: {e}")
                return False
        else:
            print(f"Trajectory 파일을 찾을 수 없습니다: {trajectory_file}")
        
        return True

def main():
    parser = argparse.ArgumentParser(description='KITTI Tracklets Visualization')
    parser.add_argument('--base_path', type=str, 
                      default='KITTI_dataset',
                      help='KITTI 데이터셋의 기본 경로')
    parser.add_argument('--dataset_prefix', type=str,
                      default='2011_09_26_drive_0020',
                      help='데이터셋 프리픽스 (예: 2011_09_26_drive_0020)')
    parser.add_argument('--max_frames', type=int,
                      default=50,
                      help='처리할 최대 프레임 수')
    parser.add_argument('--output_dir', type=str,
                      default='results_tracklets',
                      help='결과를 저장할 디렉토리')
    parser.add_argument('--disable_multi_frame_bev', action='store_true',
                      help='다중 프레임 BEV 시각화 비활성화')
    
    args = parser.parse_args()
    
    # 다중 프레임 BEV 옵션 처리
    enable_multi_frame_bev = not args.disable_multi_frame_bev
    
    def find_latest_pose_file(dataset_prefix):
        """dataset_prefix를 기반으로 results 디렉토리에서 최신 pose 파일 찾기"""
        results_dir = Path('results') / dataset_prefix
        
        if not results_dir.exists():
            print(f"Results 디렉토리를 찾을 수 없습니다: {results_dir}")
            return None
        
        # pkl과 json 파일들 찾기
        pose_files = []
        pose_files.extend(list(results_dir.glob('*poses*.pkl')))
        pose_files.extend(list(results_dir.glob('*poses*.json')))
        
        if not pose_files:
            print(f"Pose 파일을 찾을 수 없습니다: {results_dir}")
            return None
        
        # 파일명에서 타임스탬프를 기준으로 가장 최신 파일 선택
        latest_file = max(pose_files, key=lambda x: x.stat().st_mtime)
        print(f"자동으로 찾은 pose 파일: {latest_file}")
        return latest_file
    
    try:
        # 출력 디렉토리 생성
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Tracklets 시각화기 초기화
        visualizer = KITTITrackletsVisualizer(args.base_path, args.dataset_prefix)
        
        # GT 데이터 로드 (포인트클라우드와 trajectory)
        gt_loaded = visualizer.load_gt_data(args.dataset_prefix)
        if not gt_loaded:
            print("GT 데이터 로드 실패. GT BEV는 생성되지 않습니다.")
            enable_multi_frame_bev = False
        
        # Pose 파일 자동 검색 및 로드 (GT pose 우선)
        pose_file = find_latest_pose_file(args.dataset_prefix)
        if pose_file:
            pose_loaded = visualizer.load_ego_poses(str(pose_file))
            if not pose_loaded:
                print("Ego pose 로드 실패. GT BEV는 생성되지 않습니다.")
                enable_multi_frame_bev = False
        else:
            print("Ego pose 파일을 찾을 수 없습니다. GT BEV는 생성되지 않습니다.")
            enable_multi_frame_bev = False
        
        print("KITTI Tracklets Visualization 시작...")
        print(f"데이터셋: {args.dataset_prefix}")
        print(f"2D + BEV 결합 이미지: ✓")
        print(f"GT 기반 전체 프레임 BEV: {'활성화' if enable_multi_frame_bev else '비활성화'}")
        if pose_file and enable_multi_frame_bev:
            print(f"사용된 pose 파일: {pose_file}")
        if gt_loaded and enable_multi_frame_bev:
            print(f"GT 포인트클라우드: ✓")
            print(f"GT trajectory: {'✓' if visualizer.trajectory is not None else '✗'}")
        
        # 시퀀스 처리 (항상 결합된 이미지 생성)
        visualizer.process_sequence(0, args.max_frames, True, enable_multi_frame_bev)
        
        # 결과 저장
        results_file = visualizer.save_results(args.output_dir)
        
        print(f"\n완료! 결과가 {args.output_dir}/{args.dataset_prefix} 디렉토리에 저장되었습니다.")
        print(f"  • 2D + BEV 결합 이미지: ✓")
        print(f"  • GT 기반 전체 프레임 BEV: {'✓' if enable_multi_frame_bev else '✗'}")
        
    except FileNotFoundError as e:
        print(f"\n에러: {e}")
        print("\n데이터셋 구조가 다음과 같은지 확인해주세요:")
        print(f"""
KITTI_dataset/
├── tracklets/
│   └── {args.dataset_prefix}_sync/
│       └── tracklet_labels.xml
└── data/
    └── {args.dataset_prefix}_sync/
        ├── image_02/
        │   └── data/
        │       └── *.png
        └── velodyne_points/
            └── data/
                └── *.bin
        """)
    except Exception as e:
        print(f"\n예상치 못한 에러가 발생했습니다: {e}")

if __name__ == "__main__":
    main() 