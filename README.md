# 🏥 DeepSeek-R1 医疗诊断大模型微调指南

让AI成为你的「数字听诊器」！本项目带你在医疗推理领域玩转大模型微调，全程高能⚡

## 🌟 项目亮点

- 🚀 **闪电训练**：基于Unsloth框架，提速2倍显存节省80%！
- 🧠 **医学思维链**：引入CoT推理让诊断过程「透明化」
- 💊 **专业领域适配**：专治模型「医学知识贫血症」
- 📊 **训练可视化**：wandb实时监控训练过程，效果看得见
- 🎯 **精准微调**：LoRA技术实现「外科手术式」参数调整

## 🛠️ 快速开始

### 环境配置

```bash
# 创建魔法训练环境 ✨
sudo apt install python3-venv
python3 -m venv unsloth
source unsloth/bin/activate

# 安装咒语材料 📦
pip install unsloth wandb python-dotenv
```

### 启动微调

```bash
# 念动咒语启动训练！ 🔮
python r1-finetuning-unsloth.py
```

## 📈 训练过程可视化

```python
# 在wandb中查看训练数据仪表盘 📊
wandb.init(project='Fine-tune-DeepSeek-R1')
```
![image](https://github.com/user-attachments/assets/53133d3d-5b34-4e17-bb0b-03dbfd4a5d8e)

## 🧪 效果对比

### 微调前

```text
"建议多喝水，注意休息..." 🤒
```

### 微调后

```text
"根据病毒性感冒的典型病程：
1. 退烧药仅对症处理...
2. 推荐使用奥司他韦...
3. 需密切观察..." 💊
```

## 🗂️ 数据集

```python
# 医学推理黄金数据集 🏆
load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT")
```

- 500+ 中文医疗场景问答
- 包含详细思维链标注
- 覆盖常见疾病诊断路径

## ⚙️ 技术配置

| 组件     | 配置                          | 说明           |
| -------- | ----------------------------- | -------------- |
| **模型** | DeepSeek-R1-Distill-Qwen-1.5B | 医学知识蒸馏版 |
| **LoRA** | r=16, alpha=16                | 精准参数调整   |
| **量化** | 4bit 加载                     | 显存优化黑科技 |
| **训练** | BF16混合精度                  | 速度精度双保障 |

## 📦 模型保存

```python
# 保存你的医学专家模型 👩⚕️
model.save_pretrained_merged("My_Medical_GPT", save_method="merged_16bit")
```

## 🌍 模型部署

```python
# 上传到HuggingFace Hub 🌐
model.push_to_hub_merged("YourName/Medical-R1")
```

## 📌 注意事项

1. 🔑 使用前记得替换代码中的`hf_token`和`wb_token`
2. 🧪 建议先在500条数据上试跑，再扩展数据集
3. ⚠️ 医疗内容仅供参考，实际应用需专业审核

---

> 🎯 项目目标：打造「会思考」的医疗AI助手  
> 💡 小贴士：试试在wandb里对比不同LoRA参数的效果！  
> 📧 问题反馈：你的[GitHub Issue]就是我们进步的阶梯！  

![Keep Learning](https://img.shields.io/badge/-%F0%9F%93%9A_Keep_Learning!-brightgreen)
