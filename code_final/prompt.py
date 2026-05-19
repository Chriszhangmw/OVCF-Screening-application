import os
import json
import time
import requests
import base64
import mimetypes
import subprocess


API_KEY = os.environ.get("DASHSCOPE_API_KEY", "").strip()
MEDIA_DIR_ROOT = r"D:\OVCF_Test\data(3)\lost"   # 所有患者数据的根目录
OUTPUT_DIR_ROOT = r"D:\OVCF_Test\output(3)\control_group"
FFMPEG_BIN = r"D:\ffmpeg-8.0-full_build\ffmpeg-8.0-full_build\bin\ffmpeg.exe"

# --- 文件名对应 prompt ---
ACTION_TO_PROMPT_FILENAME = {
    "frontal.jpg": "image_frontal.txt",
    "lateral.jpg": "image_lateral.txt",
    "back.jpg": "image_back.txt",
    "supine_to_sit.mp4": "video_supine_to_sit.txt",
    "sit_to_supine.mp4": "video_sit_to_supine.txt",
    "roll_left.mp4": "video_roll.txt",
    "roll_right.mp4": "video_roll.txt"
}

# --- PROMPTS_CONTENT 与之前一样，省略这里示例 ---
PROMPTS_CONTENT = {
    "image_frontal.txt": """
你是一名骨科医生，熟悉骨质疏松性椎体压缩骨折（OVCF）的体态表现。
【输入】一张人体标准站立位脊柱姿态的正位照片（面对相机拍摄）。该图像已通过 OpenPose 标注人体骨骼关键点。
请基于关键点分布、身体对称性及整体平衡，从以下指标进行评估。每项以 0–1评分（0为正常，1为严重异常），并给出简短解释。
输出指标及定义：
1. 脊柱正中对称性（SpinalAlignment）：评估 C7–S1 连线相对于身体几何中线的偏移程度，反映整体脊柱是否存在侧偏或单侧塌陷。
评分标准（0–1连续量表，参考锚点如下）：
0.00 = C7–S1连线完全位于身体中线
0.10 = 极轻微偏移（<0.5cm），需仔细观察
0.20 = 轻微可见偏移（约0.5–1cm），无明显代偿
0.30 = 轻度侧偏（1–1.5cm），站姿仍稳定
0.40 = 轻中度偏移（约1.5–2cm），可能伴轻度肩或骨盆代偿
0.50 = 中度侧移（2–3cm），躯干轴线明显偏离
0.60 = 中重度偏移（3–4cm），伴明显姿势调整
0.70 = 重度侧移（>4cm），躯干倾斜明显
0.80 = 严重侧弯趋势，需代偿维持直立
0.90 = 极严重偏移，重心明显不稳
1.00 = 最严重，无法维持冠状位平衡
2. 肩部水平对称性（ShoulderSymmetry）：评估左右肩峰（Acromion）关键点高度差，反映胸椎旋转或代偿性抬肩。
评分标准（0–1连续量表）：
0.00 = 双肩完全等高，误差<0.5cm
0.10 = 极轻微高度差，仅精细测量可见
0.20 = 轻微不平（约0.5–0.8cm）
0.30 = 轻度不平（0.8–1cm），无明显旋转
0.40 = 轻中度差异（约1–1.5cm）
0.50 = 中度不平（1.5–2cm），外观明显
0.60 = 明显差异（2–2.5cm），伴胸廓旋转
0.70 = 重度抬肩（2.5–3cm），姿态异常
0.80 = 严重不平，明显结构性改变
0.90 = 极严重肩部倾斜
1.00 = 最严重，明显畸形
3. 骨盆平衡度（PelvicLeveling）：评估左右髂嵴（或ASIS对应关键点）高度差，反映骨盆倾斜与代偿性负重。
评分标准（0–1连续量表）：
0.00 = 双侧髂嵴完全水平
0.10 = 极轻微倾斜（<0.5cm）
0.20 = 轻微倾斜（0.5–1cm）
0.30 = 轻度倾斜（1–1.5cm）
0.40 = 轻中度倾斜（约1.5cm）
0.50 = 中度倾斜（1.5–2cm）
0.60 = 中重度倾斜（2–2.5cm），伴腰部代偿
0.70 = 重度骨盆侧倾（2.5–3cm）
0.80 = 严重倾斜，影响站姿稳定
0.90 = 极严重骨盆失衡
1.00 = 最严重，结构性骨盆畸形
4. 脊柱侧弯程度（CoronalCurvature）：评估冠状面上脊柱关键点（C7至S1）形成的横向弧形偏移幅度及S形曲率趋势。
评分标准（0–1连续量表）：
0.00 = 完全直线，无横向弯曲
0.10 = 极轻微曲度（<0.5cm弧度）
0.20 = 轻微单侧弧度（约0.5–1cm）
0.30 = 轻度弯曲（1–1.5cm），无明显S形
0.40 = 轻中度弯曲（1.5–2cm）
0.50 = 中度弯曲（2–3cm），出现S形趋势
0.60 = 明显S形（3cm左右）
0.70 = 重度S形弯曲
0.80 = 严重侧弯，结构性改变明显
0.90 = 极严重畸形
1.00 = 最严重，明显结构性侧弯
5. 整体平衡（GlobalBalance）：评估头部中心、C7、骨盆中心与双足中点是否位于同一垂直轴线上，反映整体重心稳定性。
评分标准（0–1连续量表）：
0.00 = 头-脊柱-骨盆-足底完全位于同一垂线
0.10 = 极轻微偏移（<0.5cm）
0.20 = 轻微重心偏移（0.5–1cm）
0.30 = 轻度失衡（1–1.5cm）
0.40 = 轻中度偏移（约1.5–2cm）
0.50 = 中度失衡（2–3cm）
0.60 = 明显偏移（3cm左右），伴代偿姿势
0.70 = 重度失衡，躯干明显偏向一侧
0.80 = 严重失衡，需要姿势调整维持站立
0.90 = 极严重不稳
1.00 = 最严重，无法独立维持平衡
请根据图像实际表现精确评分，不要四舍五入到上述锚点， 例如介于0.20和0.30之间的表现应评为0.24或0.27等。
输出格式要求：
请以标准 JSON 输出，每个指标包含分数（score），score必须有一定区分度，和一句简短的解释（explanation），请注意，这个score分数会用于后期机器学习模型学习训练，因此务必保证这个score具备临床意义，足够反应该输入临床图片或视频。
示例输出：
示例输出1（姿态基本正常的受试者）：
{
"SpinalAlignment": {"score": 0.12, "explanation": "C7-S1连线轻度偏离身体中线约0.4cm"},
"ShoulderSymmetry": {"score": 0.18, "explanation": "左肩较右侧高约0.6cm，轻微不平"},
"PelvicLeveling": {"score": 0.09, "explanation": "双侧髂嵴高度差约0.3cm，基本水平"},
"CoronalCurvature": {"score": 0.14, "explanation": "脊柱关键点呈轻微单侧弧度，横向偏移不足1cm"},
"GlobalBalance": {"score": 0.16, "explanation": "头部与骨盆中心略偏离垂直轴约0.7cm"}
}
示例输出2（存在明显冠状位失衡的受试者）：
{
"SpinalAlignment": {"score": 0.68, "explanation": "C7-S1连线向左偏移约3.6cm，躯干明显侧移"},
"ShoulderSymmetry": {"score": 0.74, "explanation": "右肩较左侧高约2.6cm，伴胸廓旋转"},
"PelvicLeveling": {"score": 0.57, "explanation": "左侧髂嵴低约1.9cm，存在明显骨盆侧倾"},
"CoronalCurvature": {"score": 0.72, "explanation": "脊柱关键点形成明显S形弯曲，最大横向偏移约3.5cm"},
"GlobalBalance": {"score": 0.63, "explanation": "头部与骨盆中心明显偏离垂直轴约3cm，重心失衡明显"}
}
【重要约束】
- 你的评分将用于机器学习训练，评分的区分度直接决定模型性能
- 每个指标必须独立评估，不同指标的分数应反映各自的实际观察，而非给出相似的分数
- 仔细观察图像中的每一个细节，即使差异很小也应体现在分数中
- 评分精度要求到小数点后两位（如0.17、0.43），禁止四舍五入到0.1或0.05的整数倍
- 如果四个指标的分数差异小于0.1，请重新审视图像，确认是否遗漏了细微差异
注意：只返回 JSON，不要添加任何额外文字。""",
    "image_back.txt": """
你是一名脊柱外科医生，熟悉背部肌肉张力与姿势代偿。
【输入】一张人体标准站立位脊柱姿态的背位照片（从背后拍摄）。该图像已通过 OpenPose 标注骨骼关键点。
请评估以下指标，每项以 0–1评分（0为正常，1为异常），并提供解释。
输出指标及定义：
1. 脊柱垂直性（SpinalVerticality）：C7 棘突至 S1 棘突连线相对于重力垂线的偏移程度，用以判断躯干整体冠状位力线是否偏离中轴；
评分标准（0–1连续量表，以下为参考锚点，允许并鼓励使用中间值）：
0.00 = 完全正常：C7–S1连线与垂线完全重合，无任何可见偏移
0.10 = 极轻微：仅在仔细观察后才可注意到的微小偏移（<0.5cm）
0.20 = 轻微可见：站立位可见微小偏移（0.5–1cm），无代偿
0.30 = 轻度异常：明确的轻度偏移（1–1.5cm），无代偿动作
0.40 = 轻中度：偏移约1.5–2cm，可能伴极轻微肩部代偿
0.50 = 中度：偏移约2–3cm，伴可见的骨盆或肩部代偿
0.60 = 中重度：偏移约3–4cm，代偿动作明确
0.70 = 重度：偏移>4cm，明显躯干倾斜
0.80 = 严重：需主动代偿才能维持站立
0.90 = 极严重：站立明显不稳
1.00 = 最严重：无法独立维持直立位
2. 肩胛骨对称性（ScapularSymmetry）：评估双侧肩胛骨下角高度及内侧缘位置是否对称，反映胸椎旋转或侧弯导致的肩部不对称情况；
评分标准（0–1连续量表，以下为参考锚点，允许并鼓励使用中间值）：
0.00 = 完全对称：双侧肩胛骨下角高度一致，误差 <0.5 cm，无任何可见差异
0.10 = 极轻微差异：需仔细观察才能发现极小高度差（约0.5 cm以内）
0.20 = 轻微可见：单侧略抬高（约0.5–0.8 cm），站姿基本可自行调整
0.30 = 轻度不对称：单侧抬高约0.5–1 cm，姿势调整后可部分改善
0.40 = 轻中度差异：高度差约1 cm，外观可见轻度肩部不平
0.50 = 中度不对称：高度差约1–1.5 cm，稳定存在，轻度胸椎旋转表现
0.60 = 明显不对称：高度差约1–2 cm，伴胸椎旋转或轻度结构性侧弯
0.70 = 重度不对称：高度差接近2 cm，肩部明显倾斜
0.80 = 严重：高度差>2 cm，明显肩部不平衡，伴胸廓旋转
0.90 = 极严重：肩部严重倾斜，姿势明显异常
1.00 = 最严重：显著结构性畸形，明显胸椎侧弯或旋转畸形
3. 骨盆旋转（PelvicRotation）：评估双侧髂前上棘（ASIS）是否位于同一冠状平面，用以判断骨盆是否存在旋转畸形；
评分标准（0–1连续量表，以下为参考锚点，允许并鼓励使用中间值）：
0.00 = 完全对称：双侧ASIS对称，骨盆正对前方，无任何旋转
0.10 = 极轻微旋转：需仔细观察才可见极小前后差异（<0.5 cm）
0.20 = 轻微旋转：单侧略前移（约0.5 cm），站姿可自行调整
0.30 = 轻度旋转：单侧前移<1 cm，姿势调整后可恢复
0.40 = 轻中度旋转：前后差约1 cm，外观可见轻度腰部代偿
0.50 = 中度旋转：差约1–1.5 cm，稳定存在，伴腰部肌肉代偿
0.60 = 中度偏重：差约1–2 cm，明显代偿姿势
0.70 = 重度旋转：接近2 cm差异，躯干出现明显旋转趋势
0.80 = 严重：>2 cm，骨盆明显扭转
0.90 = 极严重：骨盆严重扭转，站姿明显异常
1.00 = 最严重：明显结构性旋转畸形，影响站立稳定性或步态
4. 竖脊肌张力不对称（ParaspinalTensionAsymmetry）：评估双侧竖脊肌张力及肌腹隆起是否存在明显差异，反映肌肉代偿与单侧负重情况；评分标准（0–1连续量表，以下为参考锚点，允许并鼓励使用中间值）：
0.00 = 完全对称：双侧肌肉张力一致，无隆起或轮廓差异
0.10 = 极轻微差异：需仔细观察才可见轻微紧张差
0.20 = 轻微单侧紧张：触诊略紧，外观差异不明显
0.30 = 轻度不对称：单侧轻度紧张或略隆起
0.40 = 轻中度差异：外观可见轻微肌腹轮廓差
0.50 = 中度不对称：单侧隆起明显，轻度压痛
0.60 = 明显单侧紧张：肌腹隆起差异明显，伴轻度压痛
0.70 = 重度不对称：明显单侧持续紧张
0.80 = 严重：持续性单侧肌肉高张或痉挛
0.90 = 极严重：明显体表轮廓差异，姿势受影响
1.00 = 最严重：严重肌肉失衡，明显功能受限
请根据图像实际表现精确评分，不要四舍五入到上述锚点，
例如介于0.20和0.30之间的表现应评为0.24或0.27等。
输出格式要求：
请以标准 JSON 输出，每个指标包含 score 和 explanation，score必须有一定的区分度，请注意，这个score分数会用于后期机器学习模型学习训练，因此务必保证这个score具备临床意义，足够反应该输入临床图片或视频。
示例输出：
示例输出1（姿态基本正常的受试者）：
{
  "SpinalVerticality": {"score": 0.08, "explanation": "C7-S1连线几乎完全垂直，偏移<0.3cm"},
  "ScapularSymmetry": {"score": 0.15, "explanation": "双侧肩胛骨下角高度差约0.3cm，基本对称"},
  "PelvicRotation": {"score": 0.05, "explanation": "ASIS双侧对称，骨盆无旋转"},
  "ParaspinalTensionAsymmetry": {"score": 0.12, "explanation": "双侧竖脊肌轮廓基本对称"}
}
示例输出2（存在明显脊柱异常的受试者）：
{
  "SpinalVerticality": {"score": 0.65, "explanation": "C7-S1连线向右偏移约3cm，伴肩部向左代偿"},
  "ScapularSymmetry": {"score": 0.72, "explanation": "右侧肩胛骨下角较左侧高约1.8cm，胸廓可见旋转"},
  "PelvicRotation": {"score": 0.48, "explanation": "右侧ASIS略前移约1.5cm，伴腰椎轻度旋转"},
  "ParaspinalTensionAsymmetry": {"score": 0.55, "explanation": "左侧竖脊肌明显隆起，肌腹轮廓不对称"}
}
【重要约束】
- 你的评分将用于机器学习训练，评分的区分度直接决定模型性能
- 每个指标必须独立评估，不同指标的分数应反映各自的实际观察，而非给出相似的分数
- 仔细观察图像中的每一个细节，即使差异很小也应体现在分数中
- 评分精度要求到小数点后两位（如0.17、0.43），禁止四舍五入到0.1或0.05的整数倍
- 如果四个指标的分数差异小于0.1，请重新审视图像，确认是否遗漏了细微差异
注意：只返回 JSON，不要添加任何额外文字。
""",
    "image_lateral.txt": """
你是一名脊柱外科医生，擅长通过侧位片判断生理曲度变化。
【输入】一张人体标准站立位脊柱姿态的侧位照片（从侧面拍摄）。该图像已通过 OpenPose 标注骨骼关键点。
请评估以下指标，每项以 0–1评分（0为正常，1为严重异常），并提供解释。
输出指标及定义：
1. 胸椎后凸角度（ThoracicKyphosis）：评估胸椎T1–T12关键点连线形成的后凸曲度是否处于正常生理范围（约20°–40°），是否存在过度后凸或生理曲度变直。
评分标准（0–1连续量表，参考锚点如下）：
0.00 = 生理后凸正常，弧度自然
0.10 = 极轻微增大或变直（<5°偏差）
0.20 = 轻微异常（约5°–8°偏差）
0.30 = 轻度后凸增大或减小（约8°–12°）
0.40 = 轻中度异常（12°–15°）
0.50 = 中度异常（15°–20°），弧度明显改变
0.60 = 中重度后凸或明显变直
0.70 = 重度驼背或胸椎平直
0.80 = 严重结构性异常
0.90 = 极严重畸形
1.00 = 最严重，明显功能性受限
2. 腰椎前凸角度（LumbarLordosis）：评估L1–S1关键点形成的前凸曲线是否处于正常范围（约30°–50°），是否出现扁平化或反弓趋势。
评分标准（0–1连续量表）：
0.00 = 前凸弧度良好，自然生理曲线
0.10 = 极轻微减小或增大（<5°）
0.20 = 轻微异常（5°–8°）
0.30 = 轻度前凸减小或增强（8°–12°）
0.40 = 轻中度改变（12°–15°）
0.50 = 中度扁平或前凸过大（15°–20°）
0.60 = 中重度异常，腰椎弧度明显改变
0.70 = 重度扁平或反弓趋势明显
0.80 = 严重异常，腰椎结构改变明显
0.90 = 极严重异常
1.00 = 最严重，明显矢状位结构畸形
3. 躯干倾斜度（TrunkInclination）：评估C7至骨盆中心连线相对于垂直参考线的前倾角度，反映整体躯干是否前倾。
评分标准（0–1连续量表）：
0.00 = 完全直立，与垂直线重合
0.10 = 极轻微前倾（<2°）
0.20 = 轻微前倾（2°–4°）
0.30 = 轻度前倾（4°–6°）
0.40 = 轻中度前倾（6°–8°）
0.50 = 中度前倾（8°–12°）
0.60 = 中重度前倾（12°–15°）
0.70 = 重度前倾（15°以上）
0.80 = 严重前倾，需代偿维持站立
0.90 = 极严重，明显重心前移
1.00 = 最严重，无法维持正常直立
4. 头部位置偏移（HeadPosition）：评估耳垂（或外耳道）相对于肩峰垂直线的前后位置，判断是否存在头前伸姿势。
评分标准（0–1连续量表）：
0.00 = 耳垂与肩峰垂直对齐
0.10 = 极轻微前移（<0.5cm）
0.20 = 轻微前伸（0.5–1cm）
0.30 = 轻度头前伸（1–1.5cm）
0.40 = 轻中度前移（1.5–2cm）
0.50 = 中度头前伸（2–3cm）
0.60 = 明显前伸，颈椎代偿明显
0.70 = 重度头前伸姿势
0.80 = 严重前移，颈椎曲度改变明显
0.90 = 极严重前伸
1.00 = 最严重，明显姿势畸形
5. 矢状面整体平衡（SagittalBalance）：评估C7铅垂线与S1后上缘之间的水平距离（SVA），反映整体矢状位平衡状态。
评分标准（0–1连续量表）：
0.00 = C7铅垂线通过S1后上缘，完全平衡
0.10 = 极轻微偏移（<1cm）
0.20 = 轻微失衡（1–2cm）
0.30 = 轻度失衡（2–3cm）
0.40 = 轻中度失衡（3–4cm）
0.50 = 中度失衡（4–5cm）
0.60 = 中重度失衡（5–6cm）
0.70 = 重度失衡（>6cm）
0.80 = 严重失衡，重心明显前移
0.90 = 极严重失衡
1.00 = 最严重，无法维持矢状位平衡
输出格式要求：
请以标准 JSON 输出，每个指标包含 score 和 explanation，score必须有一定的区分度，请注意，这个score分数会用于后期机器学习模型学习训练，因此务必保证这个score具备临床意义，足够反应该输入临床图片或视频。
示例输出：
示例输出1（矢状位基本正常）：
{
"ThoracicKyphosis": {"score": 0.18, "explanation": "胸椎后凸略增约5°，仍接近生理范围"},
"LumbarLordosis": {"score": 0.22, "explanation": "腰椎前凸轻度减小约7°"},
"TrunkInclination": {"score": 0.16, "explanation": "躯干前倾约3°，接近直立"},
"HeadPosition": {"score": 0.21, "explanation": "耳垂位于肩峰前方约0.8cm"},
"SagittalBalance": {"score": 0.19, "explanation": "C7铅垂线较S1前移约1.5cm"}
}
示例输出2（明显矢状位失衡）：
{
"ThoracicKyphosis": {"score": 0.72, "explanation": "胸椎后凸增加约25°，驼背明显"},
"LumbarLordosis": {"score": 0.64, "explanation": "腰椎前凸明显扁平化约20°"},
"TrunkInclination": {"score": 0.69, "explanation": "躯干前倾约16°，重心明显前移"},
"HeadPosition": {"score": 0.58, "explanation": "头部前伸约2.5cm，颈椎代偿明显"},
"SagittalBalance": {"score": 0.74, "explanation": "C7铅垂线前移约6.5cm，矢状位严重失衡"}
}
【重要约束】
- 你的评分将用于机器学习训练，评分的区分度直接决定模型性能
- 每个指标必须独立评估，不同指标的分数应反映各自的实际观察，而非给出相似的分数
- 仔细观察图像中的每一个细节，即使差异很小也应体现在分数中
- 评分精度要求到小数点后两位（如0.17、0.43），禁止四舍五入到0.1或0.05的整数倍
- 如果四个指标的分数差异小于0.1，请重新审视图像，确认是否遗漏了细微差异
注意：只返回 JSON，不要添加任何额外文字。
""",
    "video_supine_to_sit.txt": """
你是一名康复与骨科专家，熟悉通过动作分析评估骨质疏松性椎体压缩骨折（OVCF）风险。
【输入】一个患者从仰卧到坐起（Supine to Sitting）的短视频，视频已通过 OpenPose 关键点标注。
请分析动作特征并从以下维度进行定量评估。每项以 0–1评分（0代表正常或流畅，1为严重异常），并提供简短解释。
输出指标及定义：
1. 动作协调性（MotionCoordination）：评估从抬头启动至完全坐稳过程中动作是否连续、节奏是否自然，是否存在分段式起身或中途停顿。
评分参考（0–1连续量表）：
0.00 = 动作连续自然
0.10 = 极轻微节奏变化
0.20 = 轻微分段但整体连贯
0.30 = 轻度停顿（<0.5秒）
0.40 = 轻中度分段完成
0.50 = 中度不连贯
0.60 = 明显分阶段完成
0.70 = 多次停顿或节奏混乱
0.80 = 严重代偿式起身
0.90 = 极严重不协调
1.00 = 无法连续完成
2. 躯干控制（TrunkControl）：评估躯干屈曲及前移过程中核心控制能力，是否存在突然前冲、惯性借力或坐起后晃动。
评分参考（0–1连续量表）：
0.00 = 屈曲平稳，控制良好
0.10 = 极轻微前移加速
0.20 = 轻微控制减弱
0.30 = 轻度前冲趋势
0.40 = 轻中度控制不足
0.50 = 中度控制不稳
0.60 = 明显借助惯性
0.70 = 重度前冲或起身后摇晃
0.80 = 严重控制缺失
0.90 = 极严重失稳
1.00 = 完全无法控制
3. 手臂支撑依赖（ArmAssistance）：评估是否使用上肢推床、撑床或牵拉以辅助坐起，反映核心力量不足或疼痛回避。
输出需包含 score 与 value 两项。
value 取值：
"无支撑"
"部分支撑"
"明显依赖支撑"
评分参考（0–1连续量表）：
0.00 = 完全无支撑
0.20左右 = 轻触床面但发力小
0.40左右 = 单手或短暂支撑
0.60左右 = 双上肢参与发力
0.80左右 = 明显依赖双臂推动
1.00 = 完全依赖上肢完成坐起
（根据支撑持续时间与发力程度进行精细评分，禁止使用固定整十值）
4. 动作完成时间（ExecutionTime）(s)：从抬头启动到完全坐稳的总耗时，结合节奏与迟疑程度评估动作效率。
评分参考（0–1连续量表）：
0.00 = 快速自然完成（<2秒）
0.10 = 略慢但节奏正常
0.20 = 轻微延长
0.30 = 轻度变慢（2–3秒）
0.40 = 轻中度缓慢
0.50 = 中度迟疑（约3–4秒）
0.60 = 明显缓慢
0.70 = 重度迟疑
0.80 = 严重缓慢或多次停顿
0.90 = 极度迟疑
1.00 = 无法独立完成
（根据视频实际时间精确量化，不得使用整十锚点）
5. 疼痛反应强度（PainResponseLevel）：观察是否存在面部紧张、屏气、动作迟疑、保护性缓慢起身等疼痛信号。
评分参考（0–1连续量表）：
0.00 = 无疼痛迹象
0.10 = 极轻微迟疑
0.20 = 轻微动作减幅
0.30 = 轻度疼痛提示
0.40 = 轻中度保护性动作
0.50 = 中度疼痛信号明显
0.60 = 明显疼痛回避行为
0.70 = 重度疼痛导致节奏改变
0.80 = 严重疼痛影响动作完成
0.90 = 极严重疼痛表现
1.00 = 因疼痛无法完成
输出格式要求：
请以标准 JSON 输出，每个指标包含 score 和 explanation，score必须有一定的区分度，请注意，这个score分数会用于后期机器学习模型学习训练，因此务必保证这个score具备临床意义，足够反应该输入临床图片或视频。
示例输出：
示例输出1（动作基本正常）：
{
"MotionCoordination": {"score": 0.22, "explanation": "动作连续，仅轻微节奏变化"},
"TrunkControl": {"score": 0.27, "explanation": "躯干屈曲平稳，控制稍减弱"},
"ArmAssistance": {"score": 0.19, "value": "部分支撑", "explanation": "单手轻触床面辅助启动"},
"ExecutionTime": {"score": 0.24, "explanation": "完成时间略延长但节奏自然"},
"PainResponseLevel": {"score": 0.21, "explanation": "动作略显谨慎，无明显疼痛表情"}
}
示例输出2（存在明显异常）：
{
"MotionCoordination": {"score": 0.74, "explanation": "动作分段完成，多次停顿"},
"TrunkControl": {"score": 0.79, "explanation": "起身过程中明显借助惯性前冲"},
"ArmAssistance": {"score": 0.86, "value": "明显依赖支撑", "explanation": "双手持续用力推床完成坐起"},
"ExecutionTime": {"score": 0.68, "explanation": "整体耗时明显延长，节奏缓慢"},
"PainResponseLevel": {"score": 0.72, "explanation": "出现保护性缓慢动作及面部紧张"}
}
【重要约束】
- 你的评分将用于机器学习训练，评分的区分度直接决定模型性能
- 每个指标必须独立评估，不同指标的分数应反映各自的实际观察，而非给出相似的分数
- 仔细观察图像中的每一个细节，即使差异很小也应体现在分数中
- 评分精度要求到小数点后两位（如0.17、0.43），禁止四舍五入到0.1或0.05的整数倍
- 如果四个指标的分数差异小于0.1，请重新审视图像，确认是否遗漏了细微差异
注意：只返回 JSON，不要添加任何额外文字。""",
    "video_sit_to_supine.txt": """
你是一名康复与骨科专家，熟悉通过动作分析评估骨质疏松性椎体压缩骨折（OVCF）风险。
【输入】一个患者从坐位回到仰卧（Sitting to Supine）的短视频，视频已通过 OpenPose 关键点标注。
请分析动作特征并从以下维度进行定量评估。每项以 0–1评分（0代表正常或流畅，1为严重异常），并提供简短解释。
输出指标及定义：
1. 动作协调性（MotionCoordination）：评估从坐位开始后移至完全仰卧过程中，动作是否连续、节奏是否自然，是否存在分段式动作或突然下落。
评分参考（0–1连续量表）：
0.00 = 动作连续平顺，无停顿
0.10 = 极轻微节奏变化
0.20 = 轻微分段但整体连贯
0.30 = 轻度节奏中断（短暂停顿<0.5秒）
0.40 = 轻中度分段完成
0.50 = 中度不连贯，存在明显停顿
0.60 = 明显分阶段下落
0.70 = 多次中断或节奏混乱
0.80 = 严重跳跃式下落
0.90 = 极严重不协调
1.00 = 无法连续完成动作
2. 躯干控制（TrunkControl）：评估躯干后移及接触床面时的离心控制能力，是否存在突然失控后倒或需要缓冲。
评分参考（0–1连续量表）：
0.00 = 平稳缓慢下落，控制良好
0.10 = 极轻微加速趋势
0.20 = 轻微控制减弱
0.30 = 轻度下落加快
0.40 = 轻中度失控趋势
0.50 = 中度控制不足
0.60 = 明显突然下落
0.70 = 重度失控，需要明显缓冲
0.80 = 严重后倒趋势
0.90 = 极严重控制缺失
1.00 = 完全失控下落
3. 手臂支撑依赖（ArmAssistance）：评估是否利用上肢支撑床面以减速或控制躯干下落，反映核心控制不足或疼痛回避。
输出需包含 score 与 value 两项。
value 取值：
"无支撑"
"部分支撑"
"明显依赖支撑"
评分参考（0–1连续量表）：
0.00 = 无手臂支撑
0.20左右 = 轻触床面但发力小
0.40左右 = 单侧明显减速
0.60左右 = 双上肢参与减速
0.80左右 = 明显依赖双臂控制下落
1.00 = 完全依赖上肢支撑完成动作
（根据支撑持续时间与发力程度给出精细评分，禁止使用固定整数倍）
4. 动作完成时间（ExecutionTime）(s)：从启动后移动作到完全仰卧接触床面的总耗时（结合节奏与迟疑程度），反映动作效率及谨慎程度。
评分参考（0–1连续量表）：
0.00 = 快速自然完成（<2秒）
0.10 = 略慢但节奏正常
0.20 = 轻微延长
0.30 = 轻度变慢（2–3秒）
0.40 = 轻中度缓慢
0.50 = 中度迟疑（约3–4秒）
0.60 = 明显缓慢
0.70 = 重度迟疑
0.80 = 严重缓慢或多次停顿
0.90 = 极度迟疑
1.00 = 无法独立完成
（根据视频实际时间精确量化，不得使用整十锚点）
5. 疼痛反应强度（PainResponseLevel）：评估动作过程中是否出现迟疑、保护性动作、缓慢控制或面部紧张等疼痛信号。
评分参考（0–1连续量表）：
0.00 = 无疼痛迹象
0.10 = 极轻微迟疑
0.20 = 轻微动作减幅
0.30 = 轻度疼痛提示
0.40 = 轻中度保护性动作
0.50 = 中度疼痛信号明显
0.60 = 明显疼痛回避行为
0.70 = 重度疼痛导致节奏改变
0.80 = 严重疼痛影响动作完成
0.90 = 极严重疼痛表现
1.00 = 因疼痛无法完成动作
输出格式要求：
请以标准 JSON 输出，每个指标包含 score 和 explanation，score必须有一定的区分度，请注意，这个score分数会用于后期机器学习模型学习训练，因此务必保证这个score具备临床意义，足够反应该输入临床图片或视频。
示例输出：
示例输出1（动作基本正常）：
{
"MotionCoordination": {"score": 0.19, "explanation": "动作连续，仅存在轻微节奏变化"},
"TrunkControl": {"score": 0.23, "explanation": "躯干下落平稳，控制稍减弱"},
"ArmAssistance": {"score": 0.18, "value": "部分支撑", "explanation": "单手轻触床面辅助末段减速"},
"ExecutionTime": {"score": 0.27, "explanation": "完成时间略延长但节奏自然"},
"PainResponseLevel": {"score": 0.21, "explanation": "动作略显谨慎，无明显疼痛表情"}
}
示例输出2（存在明显异常）：
{
"MotionCoordination": {"score": 0.71, "explanation": "动作分段完成，存在明显停顿"},
"TrunkControl": {"score": 0.78, "explanation": "躯干出现突然下落趋势，控制不足"},
"ArmAssistance": {"score": 0.84, "value": "明显依赖支撑", "explanation": "双手持续发力减速控制下落"},
"ExecutionTime": {"score": 0.69, "explanation": "整体耗时明显延长，存在迟疑"},
"PainResponseLevel": {"score": 0.73, "explanation": "出现保护性缓慢动作，提示明显疼痛"}
}
【重要约束】
- 你的评分将用于机器学习训练，评分的区分度直接决定模型性能
- 每个指标必须独立评估，不同指标的分数应反映各自的实际观察，而非给出相似的分数
- 仔细观察图像中的每一个细节，即使差异很小也应体现在分数中
- 评分精度要求到小数点后两位（如0.17、0.43），禁止四舍五入到0.1或0.05的整数倍
- 如果四个指标的分数差异小于0.1，请重新审视图像，确认是否遗漏了细微差异
注意：只返回 JSON，不要添加任何额外文字。""",
    "video_roll.txt": """
你是一名康复与骨科专家，熟悉通过动作分析评估骨质疏松性椎体压缩骨折（OVCF）风险。
【输入】一个患者进行左右翻身（Rolling Left/Right）的短视频，视频已通过 OpenPose 关键点标注。
请分析动作特征并从以下维度进行定量评估。每项以 0–1评分（0代表正常或流畅，1为严重异常），并提供简短解释。
输出指标及定义：
1. 翻身流畅性（RollingSmoothness）：评估翻身动作从启动到完成的连续性，包括角速度是否平稳、是否出现中途停滞、减速或明显分段。
评分参考（0–1连续量表）：
0.00 = 动作连续、匀速完成，无停顿
0.10 = 极轻微节奏变化，几乎不可察觉
0.20 = 轻微速度波动，但整体连贯
0.30 = 轻度节奏中断，出现短暂停顿(<0.5秒)
0.40 = 轻中度分段动作，存在明显减速
0.50 = 中度卡顿，动作分为两个阶段完成
0.60 = 明显停滞或犹豫后继续
0.70 = 多次停顿，翻身过程断续
0.80 = 严重不流畅，需要明显调整后继续
0.90 = 几乎无法连续完成动作
1.00 = 无法独立完成翻身
2. 髋肩协调性（HipShoulderCoordination）：评估肩带与骨盆带在横轴翻转过程中的同步性（相位差、旋转角度启动顺序），判断是否存在分离运动或代偿。
评分参考（0–1连续量表）：
0.00 = 肩髋同步旋转，相位差极小
0.10 = 极轻微相位差（<10°）
0.20 = 轻微不同步但整体协调
0.30 = 轻度分离运动（肩先动或髋先动）
0.40 = 轻中度相位差（约15°–20°）
0.50 = 中度分离运动，明显分段旋转
0.60 = 明显肩髋不同步
0.70 = 重度分离运动，存在躯干“块状”移动
0.80 = 严重协调障碍
0.90 = 极严重不同步，依赖代偿
1.00 = 完全失去协同机制
3. 疼痛反应（PainDuringMovement）：评估翻身过程中是否出现迟疑、突然停止、保护性收缩或面部/动作防御信号，推测疼痛干扰程度。
评分参考（0–1连续量表）：
0.00 = 无迟疑或保护性动作
0.10 = 极轻微迟疑
0.20 = 轻微放慢启动速度
0.30 = 轻度犹豫或动作减幅
0.40 = 轻中度迟疑，节奏明显改变
0.50 = 中度疼痛提示，翻身分段完成
0.60 = 明显疼痛反应，动作停顿
0.70 = 重度疼痛回避行为
0.80 = 严重疼痛导致明显动作中断
0.90 = 极严重疼痛表现
1.00 = 因疼痛无法完成翻身
4. 支撑手使用情况（SupportHandUsage）：评估翻身过程中是否使用上肢支撑床面协助启动或完成动作，反映躯干力量不足或疼痛代偿。
输出包含 score 与 value 两项。
value 取值：
"无使用"
"单手辅助"
"双手发力辅助"
评分参考（0–1连续量表）：
0.00 = 无手部支撑
0.25左右 = 单手轻度辅助（轻推）
0.50左右 = 单手明显发力
0.75左右 = 双手参与但非持续用力
1.00 = 双手明显发力推床完成翻身
（根据发力强度与持续时间给予精细评分，禁止使用0.25、0.50等固定值，需给出区分度分数）
输出格式要求：
请以标准 JSON 输出，每个指标包含 score 和 explanation，score必须有一定的区分度，请注意，这个score分数会用于后期机器学习模型学习训练，因此务必保证这个score具备临床意义，足够反应该输入临床图片或视频。
示例输出：
示例输出1（动作基本正常）：
{
"RollingSmoothness": {"score": 0.18, "explanation": "翻身过程整体连贯，仅存在轻微减速"},
"HipShoulderCoordination": {"score": 0.22, "explanation": "肩髋基本同步，存在轻微相位差"},
"PainDuringMovement": {"score": 0.16, "explanation": "启动略慢，但无明显保护性动作"},
"SupportHandUsage": {"score": 0.21, "value": "单手辅助", "explanation": "单手轻触床面辅助启动，发力较小"}
}
示例输出2（存在明显异常）：
{
"RollingSmoothness": {"score": 0.67, "explanation": "翻身过程中两次明显停顿，动作分段完成"},
"HipShoulderCoordination": {"score": 0.72, "explanation": "肩先动髋滞后，相位差明显"},
"PainDuringMovement": {"score": 0.64, "explanation": "中途停顿并出现保护性缓慢动作"},
"SupportHandUsage": {"score": 0.88, "value": "双手发力辅助", "explanation": "双手持续推床完成翻身，代偿明显"}
}
【重要约束】
- 你的评分将用于机器学习训练，评分的区分度直接决定模型性能
- 每个指标必须独立评估，不同指标的分数应反映各自的实际观察，而非给出相似的分数
- 仔细观察图像中的每一个细节，即使差异很小也应体现在分数中
- 评分精度要求到小数点后两位（如0.17、0.43），禁止四舍五入到0.1或0.05的整数倍
- 如果四个指标的分数差异小于0.1，请重新审视图像，确认是否遗漏了细微差异
注意：只返回 JSON，不要添加任何额外文字。"""
}


def convert_video_to_mp4(input_path):
    """将视频压制为 H264，适用于长视频（1 分钟以上）"""
    base, ext = os.path.splitext(input_path)
    output_path = base + "_conv.mp4"

    cmd = [
        FFMPEG_BIN, "-y",
        "-i", input_path,
        "-c:v", "libx264",
        "-crf", "25",            # ⭐ 更强压缩，适用于长视频
        "-preset", "medium",
        "-c:a", "aac",
        "-b:a", "96k",
        output_path
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"❌ 视频转码失败: {input_path}\n{e.stderr.decode()}")
        return None



def file_to_base64_data_url(file_path):
    """读取文件并转换为 base64 Data URL"""
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type:
        ext = os.path.splitext(file_path)[1].lower()
        mime_map = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.mp4': 'video/mp4'}
        mime_type = mime_map.get(ext)
        if not mime_type: return None
    with open(file_path, 'rb') as f:
        binary_data = f.read()
    b64 = base64.b64encode(binary_data).decode('utf-8')
    return f"data:{mime_type};base64,{b64}"


def call_dashscope(media_data_url, prompt_text):
    """支持长视频（1分钟以上）的 DashScope 调用"""

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    media_type = "image" if media_data_url.startswith('data:image') else "video"

    # ⭐ 重点：自动为长视频设置采样策略
    content = [
        {"text": prompt_text},
        {media_type: media_data_url}
    ]

    payload = {
        "model": "qwen-vl-max",
        "input": {
            "messages": [{
                "role": "user",
                "content": content
            }],
            # ⭐⭐ 加入视频采样策略，避免传入太大视频导致失败
            "video_frame_sampling": {
                "strategy": "uniform",
                "sample_frame_count": 16
            }
        },
        "parameters": {
            "result_format": "message"
        }
    }

    try:
        resp = requests.post(
            "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation",
            headers=headers,
            json=payload,
            timeout=600    # ⭐ 增加为 10 分钟，以处理长视频
        )
        resp.raise_for_status()
        return resp.json()

    except requests.exceptions.Timeout:
        return {"error": "请求超时（长视频需要更长时间，可调大 timeout）"}

    except requests.exceptions.RequestException as e:
        return {"error": f"请求失败: {e}"}



def process_patient(patient_folder, output_folder):
    success_count = 0
    fail_count = 0
    for action_key, prompt_filename in ACTION_TO_PROMPT_FILENAME.items():
        local_file_path = os.path.join(patient_folder, action_key)
        if not os.path.exists(local_file_path):
            continue

        print(f"\n=============== 正在处理：{action_key} ===============")

        # 视频需要转码
        if local_file_path.lower().endswith(".mp4"):
            converted = convert_video_to_mp4(local_file_path)
            if not converted:
                fail_count += 1
                continue
            media_file = converted
        else:
            media_file = local_file_path

        media_data_url = file_to_base64_data_url(media_file)
        if not media_data_url:
            fail_count += 1
            continue

        prompt_text = PROMPTS_CONTENT[prompt_filename].strip()

        result = None
        for attempt in range(3):
            result = call_dashscope(media_data_url, prompt_text)
            if result and "output" in result:
                break
            else:
                print(f"⚠️ 请求失败 (尝试 {attempt+1}/3): {result.get('error', '未知错误')}")
                time.sleep(2)

        output_filename_base = os.path.splitext(action_key)[0]
        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(output_folder, f"{output_filename_base}.json")

        if result and "output" in result and result["output"]["choices"]:
            message_content_list = result["output"]["choices"][0]["message"]["content"]
            json_string_to_parse = None
            if isinstance(message_content_list, list) and message_content_list:
                first_part = message_content_list[0]
                if isinstance(first_part, dict) and 'text' in first_part:
                    json_string_to_parse = first_part['text']
            if json_string_to_parse:
                try:
                    final_output = json.loads(json_string_to_parse)
                except json.JSONDecodeError:
                    final_output = {"raw_content": json_string_to_parse}
            else:
                final_output = {"error": "No text content", "raw_response": message_content_list}
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(final_output, f, ensure_ascii=False, indent=4)
            print(f"✅ 成功保存结果：{output_file}")
            success_count += 1
        else:
            print(f"❌ 最终处理失败：{action_key}")
            fail_count += 1
        time.sleep(1)
    print(f"\n🎉 患者处理完成！ ✅ 成功: {success_count}, ❌ 失败: {fail_count}")


def main():
    if not API_KEY or "sk-" not in API_KEY:
        print("❌ API_KEY 无效，请检查。")
        return
    if not os.path.isdir(MEDIA_DIR_ROOT):
        print(f"❌ 数据根目录不存在: {MEDIA_DIR_ROOT}")
        return

    for patient_id in os.listdir(MEDIA_DIR_ROOT):
        patient_folder = os.path.join(MEDIA_DIR_ROOT, patient_id)
        if not os.path.isdir(patient_folder):
            continue
        output_folder = os.path.join(OUTPUT_DIR_ROOT, patient_id)
        print(f"\n===== 开始处理患者: {patient_id} =====")
        process_patient(patient_folder, output_folder)


if __name__ == "__main__":
    main()
