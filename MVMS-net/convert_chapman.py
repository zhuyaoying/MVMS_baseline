"""
convert_chapman.py
------------------
将 WFDB_ChapmanShaoxing 数据集转换为与 CPSC 相同的目录结构，
以便直接用于 MVMS-net 训练。

运行方式（在 MVMS_baseline-main/ 根目录下执行）：
    cd /path/to/MVMS_baseline-main
    python MVMS-net/convert_chapman.py

输出结构：
    data/Chapman/
        records100/          # 100 Hz WFDB 记录
        records500/          # 500 Hz WFDB 记录
        chapman_database.csv # 元数据 + 标签
"""

import os
import ast
import pandas as pd
import wfdb
from tqdm import tqdm
import numpy as np
from scipy.ndimage import zoom

# ── 路径配置（相对于 MVMS_baseline-main/ 根目录）────────────────────────────
INPUT_FOLDER  = 'data/WFDB_ChapmanShaoxing/'
OUTPUT_FOLDER = 'data/Chapman/'
OUTPUT_100    = OUTPUT_FOLDER + 'records100/'
OUTPUT_500    = OUTPUT_FOLDER + 'records500/'

for folder in [OUTPUT_FOLDER, OUTPUT_100, OUTPUT_500]:
    os.makedirs(folder, exist_ok=True)

# ── SNOMED-CT 编码 → 标签缩写映射 ─────────────────────────────────────────
SNOMED_MAP = {
    '426177001': 'SB',      # Sinus Bradycardia 窦性心动过缓
    '164934002': 'TWC',     # T-Wave Change T波改变
    '426783006': 'NSR',     # Normal Sinus Rhythm 正常窦性心律
    '164889003': 'AFIB',    # Atrial Fibrillation 心房颤动
    '427084000': 'STACH',   # Sinus Tachycardia 窦性心动过速
    '55827005':  'LVH',     # Left Ventricular Hypertrophy 左室肥厚
    '428750005': 'NSSTEC',  # Nonspecific ST-T Change 非特异ST-T改变
    '426761007': 'SVT',     # Supraventricular Tachycardia 室上性心动过速
    '59118001':  'CRBBB',   # Complete Right Bundle Branch Block 完全右束支传导阻滞
    '713422000': 'CRBBB',   # Complete Right Bundle Branch Block (alt code)
    '164890007': 'AFL',     # Atrial Flutter 心房扑动
    '429622005': 'STD_',    # ST-Depression ST段压低
    '39732003':  'LAD',     # Left Axis Deviation 电轴左偏
    '17338001':  'VEB',     # Ventricular Ectopic Beats 室性异位搏动
    '284470004': 'PAC',     # Premature Atrial Contraction 房性早搏
    '251146004': 'LQRSV',   # Low QRS Voltages 低QRS电压
    '270492004': 'IAVB',    # First Degree AV Block 一度房室传导阻滞
    '698252002': 'NSIVCB',  # Nonspecific Intraventricular Conduction Block
    '164917005': 'QWAVE',   # Q-Wave Abnormality Q波异常
    '47665007':  'RAD',     # Right Axis Deviation 电轴右偏
    '164909002': 'LBBB',    # Left Bundle Branch Block 左束支传导阻滞
    '164931005': 'ISTD',    # ST-Depression (variant)
    '233917008': 'AVB',     # AV Block 房室传导阻滞
    '251199005': 'PR',      # Pacing Rhythm 起搏心律
    '59931005':  'TWI',     # T-Wave Inversion T波倒置
    '164912004': 'LAFB',    # Left Anterior Fascicular Block 左前分支传导阻滞
    '251198002': 'PRII',    # Prolonged PR Interval PR间期延长
    '111975006': 'LQT',     # Long QT Syndrome 长QT综合征
    '164865005': 'MI',      # Myocardial Infarction 心肌梗死
    '428417006': 'IRBBB',   # Incomplete Right Bundle Branch Block 不完全右束支传导阻滞
}

CHANNELS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def store_wfdb(record_name, signal, out_dir, fs):
    """将信号保存为 WFDB 格式。"""
    wfdb.wrsamp(
        record_name,
        fs=fs,
        sig_name=CHANNELS,
        p_signal=signal.astype(np.float64),
        units=['mV'] * 12,
        fmt=['16'] * 12,
        write_dir=out_dir,
    )


def assign_folds(n, n_folds=10, seed=42):
    """随机打乱后循环分配 fold 编号（1-10），无需额外依赖。"""
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(n)
    folds = np.empty(n, dtype=int)
    for rank, pos in enumerate(shuffled):
        folds[pos] = (rank % n_folds) + 1
    return folds


# ── 主转换循环 ───────────────────────────────────────────────────────────────
hea_files = sorted([f for f in os.listdir(INPUT_FOLDER) if f.endswith('.hea')])
records = []
ecg_counter = 0
skipped = 0

print(f"共找到 {len(hea_files)} 个 .hea 文件，开始转换...")

for hea_file in tqdm(hea_files):
    record_name = hea_file[:-4]  # 去掉 .hea 后缀
    record_path = os.path.join(INPUT_FOLDER, record_name)

    # ── 读取信号 ──────────────────────────────────────────────────────────
    try:
        signal, meta = wfdb.rdsamp(record_path)
    except Exception as e:
        print(f"\n跳过 {record_name}: {e}")
        skipped += 1
        continue

    # ── 解析 .hea 注释（Age / Sex / Dx）──────────────────────────────────
    age, sex, dx_codes = -1, -1, []
    try:
        header = wfdb.rdheader(record_path)
        for comment in header.comments:
            comment = comment.strip()
            if comment.startswith('Age:'):
                try:
                    age = int(comment.split(':', 1)[1].strip())
                except ValueError:
                    age = -1
            elif comment.startswith('Sex:'):
                sex_str = comment.split(':', 1)[1].strip().lower()
                sex = 1 if sex_str == 'male' else 0
            elif comment.startswith('Dx:'):
                dx_codes = [c.strip() for c in comment.split(':', 1)[1].split(',')]
    except Exception:
        pass

    # ── 映射标签 ─────────────────────────────────────────────────────────
    labels = {SNOMED_MAP[c]: 100 for c in dx_codes if c in SNOMED_MAP}
    if not labels:
        skipped += 1
        continue  # 跳过无已知标签的记录

    # ── 保证 12 导联 ──────────────────────────────────────────────────────
    if signal.shape[1] < 12:
        print(f"\n跳过 {record_name}: 导联数不足 ({signal.shape[1]})")
        skipped += 1
        continue
    signal = signal[:, :12]

    ecg_counter += 1
    out_name = str(ecg_counter)

    # ── 存储 500 Hz 记录 ──────────────────────────────────────────────────
    store_wfdb(out_name, signal, OUTPUT_500, 500)

    # ── 降采样至 100 Hz（5000 → 1000 采样点）并存储 ────────────────────
    sig_100 = zoom(signal, (0.2, 1.0))
    store_wfdb(out_name, sig_100, OUTPUT_100, 100)

    records.append({
        'ecg_id':    ecg_counter,
        'filename':  record_name,
        'validation': False,
        'age':       age,
        'sex':       sex,
        'scp_codes': labels,
    })

# ── 构建 DataFrame 并保存 CSV ────────────────────────────────────────────────
df = pd.DataFrame(records).set_index('ecg_id')
df['patient_id'] = df.index
df['strat_fold'] = assign_folds(len(df))

csv_path = OUTPUT_FOLDER + 'chapman_database.csv'
df.to_csv(csv_path)

print(f"\n转换完成！")
print(f"  成功: {len(df)} 条记录  →  {csv_path}")
print(f"  跳过: {skipped} 条记录（无匹配标签或读取失败）")
print(f"  标签类别: {sorted(set(l for s in df.scp_codes for l in ast.literal_eval(str(s)).keys()))}")
