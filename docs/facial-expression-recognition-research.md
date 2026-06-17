# 面部表情识别方法调研

本文记录 MindBridge / multimodalAgent 项目中“实时视觉情绪识别”方向的调研结果。核心结论：**SOTA 方法** 与 **工程 baseline** 必须分开看。DeepFace 这类传统工具库不作为当前路线；当前可运行静态 FER 实验选择 **POSTER++**，微表情短视频方向保留 **FED-PsyAU** 作为研究实验。

## SOTA / 论文方法

| 方法名称 | 时间 | 发表刊物/来源 | 结果摘要 | 分类标签类型 | 本地实验判断 |
|---|---:|---|---|---|---|
| MPA-FER | 2025 | ICCV 2025 | RAF-DB 93.74%，AffectNet-7 68.89%，AffectNet-8 63.74%，FERPlus 91.81% | 7/8 类离散表情；CLIP/VLM prompt 对齐 | 指标最强，但暂未找到可直接运行的公开代码/权重 |
| SynFER | 2025 | ICCV 2025 | 合成数据训练 AffectNet 67.23%，5x 合成数据 69.84% | 合成数据生成 + FER 训练增强 | 更像训练增强方法，不是开箱推理器 |
| FED-PsyAU | 2025 | ICCV 2025 | 在 CAS(ME)³、DFME 微表情数据库上取得强表现 | 微表情视频 MER；CAS(ME)³ 三分类，DFME 七分类；AU 动态关系建模 | **当前短视频实验方法**，官方 PyTorch 代码公开，但真实推理依赖上游代码、checkpoint 和光流预处理 |
| HGLFFNet | 2026 | Neurocomputing | RAF-DB 92.44%，AffectNet-7 67.83%，AffectNet-8 64.08%，FERPlus 92.21% | 7/8 类离散表情；全局 + 局部特征融合 | 论文指标强，但公开工程材料不如 POSTER 系列清晰 |
| POSTER++ | 2025 正刊 / 2023 arXiv | Pattern Recognition / arXiv | RAF-DB 92.21%，AffectNet-7 67.49%，AffectNet-8 63.77% | 7/8 类离散表情；简化 landmark 融合 | **当前静态 FER 实验方法**，官方代码和 checkpoint 下载入口可用 |
| POSTER-Var | 2026 | Scientific Reports / GitHub | RAF-DB 92.76%，AffectNet-7 67.91%，FERPlus 91.89% | 离散表情 + 不确定性建模 | **当前优先实验方法**，公开代码和 best models 下载入口可用 |
| FERMam | 2026 | Scientific Reports / GitHub | RAF-DB 92.13%，AffectNet-7 66.38%，AffectNet-8 61.45%，FERPlus 91.68% | 7/8 类离散表情；图像分支 + landmark 分支 | 思路贴近 Face Mesh，但 checkpoint 获取和实时推理还需验证 |
| AUNet | 2026 | Knowledge-Based Systems | 摘要称在 RAF-DB、AffectNet、FERPlus 上有效 | Action Unit 驱动 + 离散情绪分类 | 可解释性方向值得关注，暂不作为第一版实验 |

## 工程 Baseline / 可部署方法

| 方法名称 | 来源 | 标签类型 | 本地部署判断 |
|---|---|---|---|
| EmotiEffLib | PyPI / GitHub / 官方文档 | 7/8 类离散表情；可做 engagement | 已做隔离实验，但体感不够准，仅保留作轻量 baseline |
| HSEmotion | PyPI / GitHub | 8 类离散表情 | EmotiEffLib 前身，可快速试验，但不是 SOTA 路线 |
| LibreFace | WACV 2024 / GitHub | 表情分类 + Action Unit 检测/强度 | 可解释性强，适合后续 AU 实验 |

## 本项目推荐路线

1. **当前静态 FER 实验**：新增 `POSTER++ Lab`，跑在 `127.0.0.1:8096`，使用浏览器摄像头单帧抽图分析。
2. **当前微表情视频实验**：保留 `FED-PsyAU Lab`，跑在 `127.0.0.1:8095`，使用浏览器短视频片段而不是单帧图片。
3. **权重策略**：POSTER++ checkpoint 放在 `models/poster-plus-plus/checkpoints/`，pretrain 放在 `models/poster-plus-plus/pretrain/`；缺少 checkpoint、pretrain 或 upstream 时接口明确返回 503。
4. **正式接入前**：保持实验输出与正式 Python 后端 `MEDIAPIPE_MODE=http` 响应语义兼容，包括 `emotion`、`visualEmotion`、`score`、`visualScore`、`riskLevel`、`confidence`、`evidence`、`features`。
5. **产品语义**：面部表情只能作为弱视觉信号，不能直接等同于心理状态或医学诊断；报告链路仍应以文本、音频、视觉融合评估为主。

## POSTER++ 本地实验要点

- Upstream：`https://github.com/Talented-Q/POSTER_V2`
- 论文：`https://www.sciencedirect.com/science/article/pii/S0031320324007027`
- 任务：普通静态面部表情识别（FER），适合摄像头抽帧。
- 默认 checkpoint：`models/poster-plus-plus/checkpoints/rafdb_best.pth`
- 默认 pretrain：`models/poster-plus-plus/pretrain/ir50.pth`、`models/poster-plus-plus/pretrain/mobilefacenet_model_best.pth.tar`
- 默认 RAF-DB 7 类标签：`surprise`、`fear`、`disgust`、`happy`、`sad`、`angry`、`neutral`。
- 实验模块不保存图片帧、不写数据库、不触发报告、Excel 或预警。

## FED-PsyAU 本地实验要点

- Upstream：`https://github.com/MELABIPCAS/FED-PsyAU`
- 论文：`https://openaccess.thecvf.com/content/ICCV2025/html/Li_FED-PsyAU_Privacy-Preserving_Micro-Expression_Recognition_via_Psychological_AU_Coordination_and_Dynamic_ICCV_2025_paper.html`
- 任务：微表情识别（MER），不是普通静态表情识别（FER）。
- 输入：短视频片段或 onset/apex 相关动态特征；不适合 canvas 单帧抽图。
- CAS(ME)³ 标签：`positive`、`negative`、`surprise`。
- DFME 标签：`Happiness`、`Surprise`、`Disgust`、`Sadness`、`Anger`、`Fear`、`Contempt`。
- 真实推理依赖上游代码、checkpoint、TV-L1 光流、ROI/AU 预处理；第一版实验先提供短视频入口和缺依赖提示。

## POSTER-Var 本地实验要点

- Upstream：`https://github.com/lg2578/poster-var`
- 论文：`https://pmc.ncbi.nlm.nih.gov/articles/PMC12923884/`
- 权重：作者 README 指向 best models 下载入口；如需网盘人工下载，放入 `models/poster-var/checkpoints/rafdb_best.pth`。
- 默认 RAF-DB 7 类标签：`surprise`、`fear`、`disgust`、`happy`、`sad`、`angry`、`neutral`。
- 实验模块不保存图片帧、不写数据库、不触发报告、Excel 或预警。

## 参考来源

- [POSTER-Var GitHub](https://github.com/lg2578/poster-var)
- [POSTER-Var Scientific Reports / PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12923884/)
- [POSTER GitHub](https://github.com/zczcwh/POSTER)
- [POSTER++ arXiv](https://arxiv.org/abs/2301.12149)
- [MPA-FER ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Ma_Multimodal_Prompt_Alignment_for_Facial_Expression_Recognition_ICCV_2025_paper.html)
- [SynFER ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/He_SynFER_Towards_Boosting_Facial_Expression_Recognition_with_Synthetic_Data_ICCV_2025_paper.html)
- [HGLFFNet Neurocomputing 2026](https://www.sciencedirect.com/science/article/abs/pii/S0925231225027687)
- [FERMam GitHub](https://github.com/jxcsglr/FERMam)
- [AUNet Knowledge-Based Systems](https://www.sciencedirect.com/science/article/pii/S0950705126003114)
- [FED-PsyAU ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Li_FED-PsyAU_Privacy-Preserving_Micro-Expression_Recognition_via_Psychological_AU_Coordination_and_Dynamic_ICCV_2025_paper.html)
- [FED-PsyAU GitHub](https://github.com/MELABIPCAS/FED-PsyAU)
