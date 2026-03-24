import os

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForMaskedLM

from mld.utils.repo_paths import resolve_stuxlm_pooler_ckpt, resolve_transformers_cache_dir


def load_StuXLM(kd_version, model):
    ckpt_path = resolve_stuxlm_pooler_ckpt(kd_version)
    state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    model.load_state_dict(state_dict)
    print(f"StuXLM load from [{ckpt_path}]")


class StudentModel(nn.Module):
    """XLM-R 学生模型（带可选的 pooler 与蒸馏权重）。"""

    def __init__(
        self,
        model_name: str = "FacebookAI/xlm-roberta-base",
        kd_version: str = "KD_BIKL_CEH3D_20",
        cache_dir: str | None = None,
        use_student_pooler: bool = True,
    ):
        super().__init__()
        if cache_dir is None:
            cache_dir = resolve_transformers_cache_dir()
        os.makedirs(cache_dir, exist_ok=True)
        self.xlm = AutoModelForMaskedLM.from_pretrained(
            model_name, cache_dir=cache_dir, ignore_mismatched_sizes=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        # 与历史实现一致：外部传入的开关曾被覆盖为 True，此处保持相同行为以保证可复现
        use_student_pooler = True
        self.use_student_pooler = use_student_pooler
        if use_student_pooler:
            self.pooler = nn.Sequential(nn.LayerNorm(768, eps=1e-12), nn.Linear(768, 512))
            load_StuXLM(kd_version, self)
        self.device = None

    def forward(self, text):
        if self.device is None:
            self.device = self.xlm.roberta.embeddings.LayerNorm.bias.device
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.xlm(**inputs, return_dict=True, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]
            if self.use_student_pooler:
                cls_token = self.pooler(last_hidden[:, 0])
            else:
                cls_token = last_hidden[:, 0]
            text_length = inputs["attention_mask"].sum(dim=1).tolist()
            cls_token = cls_token.unsqueeze(1)
            return text_length, last_hidden, cls_token
