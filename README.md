# RGB–T 视频微小目标检测研究项目

面向跨模态错位的可靠性引导 RGB–T 视频微小目标检测。当前包含研究方案、开发流程、模型框架图、参考论文及模块代码；完整模型尚未实现和训练。

## 项目入口

- [研究方案](研究方案_未对齐RGBT视频微小目标检测.md)
- [开发工作流程与实验计划](开发工作流程_完整模型与消融实验.md)
- [框架图说明](模型框架图/README_图稿说明.md)
- `模型框架图/`：总体架构及 OCR、TPSI、RGCM 的 SVG、PNG 和 AI 草图。
- `模型整体框架图范例/`、`模块框架图范例/`：参考图。
- 各论文文件夹：论文 PDF 及参考模块代码；保留原作者权利与许可，不构成本项目原创实现。

## 在另一台电脑继续

先登录有仓库访问权限的 GitHub 账号，再执行：

```bash
gh auth login
gh repo clone zse04152005-del/rgbt-video-tiny-research
cd rgbt-video-tiny-research
```

每次开始工作前执行 `git pull --ff-only`。完成修改后：

```bash
git add .
git commit -m "Update research project"
git push
```

切换电脑前先提交并推送，另一台电脑拉取后继续。数据集和训练环境后续按开发流程准备。
