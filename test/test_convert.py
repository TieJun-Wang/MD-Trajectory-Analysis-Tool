import sys 
from pathlib import Path
import numpy as np
import re
this_file = Path(__file__).resolve()
root = this_file.parent.parent
sys.path.append(str(root))
from mdta import ms_structure


xsd_path = "C:\\temp\MDT\\122C12-450K.xsd"
trr_path = "C:\\temp\MDT\\122C12-450K.trr"
xtc_path = ""
refer_path = "C:\\temp\MDT\\122C12-450K.txt"
 
def read_ms_trj_ascii(file_path):
    all_frames = []
    stop_keys = ["Pressure:", "Canonical", "DefCell:", "Period:", "Rwp:", "Velocities:", "Forces:"]
    pat = re.compile(r'\s*(\d+)\s+([+-]?\d+\.\d+e[+-]\d+)\s+([+-]?\d+\.\d+e[+-]\d+)\s+([+-]?\d+\.\d+e[+-]\d+)')

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 按 Coordinates: 分割块
    blocks = re.split(r'Coordinates:', content)
    for blk in blocks[1:]:
        # 找到第一个标记，截断
        first_stop_pos = len(blk)
        for k in stop_keys:
            pos = blk.find(k)
            if pos != -1 and pos < first_stop_pos:
                first_stop_pos = pos
        coord_text = blk[:first_stop_pos]
        matches = pat.findall(coord_text)
        frame_coords = []
        for m in matches:
            idx = int(m[0])
            x = float(m[1])
            y = float(m[2])
            z = float(m[3])
            frame_coords.append([x,y,z])
        all_frames.append(frame_coords)
    return all_frames

u = ms_structure.build_universe(ms_structure.read_ms_document(xsd_path))

u.load_new(trr_path)   

refer_info = read_ms_trj_ascii(refer_path)

mean_list = []

for i in range(len(refer_info)-1):

    refer_position = np.array(refer_info[i])
    data_position = u.trajectory[i]

    diff = data_position - refer_position
    abs_diff = abs(diff)

    # 全局最大偏差（所有原子所有xyz里面最大那一个）
    max_abs_diff = np.max(abs_diff)
    # 平均绝对偏差
    mean_abs_diff = np.mean(abs_diff)
    mean_list.append(mean_abs_diff)
    print(f'帧数:{i+1}')
    print(f"坐标全局最大绝对偏差 (Å): {max_abs_diff:.8f}")
    print(f"坐标平均绝对偏差 (Å): {mean_abs_diff:.8f}")

avg_diff = sum(mean_list)/len(mean_list)
print("=============================================================")
print(f"\033[32m统计总帧数:{len(refer_info)-1}\033[0m")
print(f"\033[32m单帧原子数:{u.atoms.n_atoms}\033[0m")
print(f"\033[32m统计总原子数:{(len(refer_info)-1)*u.atoms.n_atoms}\033[0m")
print(f"\033[32m平均单原子绝对偏差 (Å):{avg_diff:.8f}\033[0m")
