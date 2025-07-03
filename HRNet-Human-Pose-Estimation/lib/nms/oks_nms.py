# oks_nms.py
# COCO-style OKS 기반 NMS (임시 버전)
# HRNet test_cpu.py가 정상 작동하게 하기 위한 더미 함수입니다.

import numpy as np

def oks_nms(kpts_db, thresh, sigmas=None, in_vis_thre=None):
    """
    임시 OKS NMS 함수.
    입력된 keypoints 리스트의 인덱스를 그대로 반환합니다.

    Parameters:
    - kpts_db (list): 사람 keypoints 정보가 담긴 리스트
    - thresh (float): OKS threshold (무시됨)
    - sigmas (list): 관절별 sigma 값 (무시됨)
    - in_vis_thre (float): 시각 임계값 (무시됨)

    Returns:
    - keep (list of int): 사용할 인덱스 (전부 다 유지)
    """
    return list(range(len(kpts_db)))  # 모두 유지
