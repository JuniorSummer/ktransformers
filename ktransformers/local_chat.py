"""
Description  :  
Author       : Boxin Zhang, Azure-Tang
Version      : 0.1.0
Copyright (c) 2024 by KVCache.AI, All Rights Reserved. 
"""

import os
import platform
import sys

project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_dir)
import torch
import logging
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    GenerationConfig,
    TextStreamer,
)
import json
import fire
from ktransformers.optimize.optimize import optimize_and_load_gguf
from ktransformers.models.modeling_deepseek import DeepseekV2ForCausalLM
from ktransformers.models.modeling_qwen2_moe import Qwen2MoeForCausalLM
from ktransformers.models.modeling_deepseek_v3 import DeepseekV3ForCausalLM
from ktransformers.models.modeling_llama import LlamaForCausalLM
from ktransformers.models.modeling_mixtral import MixtralForCausalLM
from ktransformers.util.utils import prefill_and_generate
from ktransformers.server.config.config import Config
from ktransformers.operators.flashinfer_wrapper import flashinfer_enabled
import psutil
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlShutdown


custom_models = {
    "DeepseekV2ForCausalLM": DeepseekV2ForCausalLM,
    "DeepseekV3ForCausalLM": DeepseekV3ForCausalLM,
    "Qwen2MoeForCausalLM": Qwen2MoeForCausalLM,
    "LlamaForCausalLM": LlamaForCausalLM,
    "MixtralForCausalLM": MixtralForCausalLM,
}

ktransformer_rules_dir = (
    os.path.dirname(os.path.abspath(__file__)) + "/optimize/optimize_rules/"
)
default_optimize_rules = {
    "DeepseekV2ForCausalLM": ktransformer_rules_dir + "DeepSeek-V2-Chat.yaml",
    "DeepseekV3ForCausalLM": ktransformer_rules_dir + "DeepSeek-V3-Chat.yaml",
    "Qwen2MoeForCausalLM": ktransformer_rules_dir + "Qwen2-57B-A14B-Instruct.yaml",
    "LlamaForCausalLM": ktransformer_rules_dir + "Internlm2_5-7b-Chat-1m.yaml",
    "MixtralForCausalLM": ktransformer_rules_dir + "Mixtral.yaml",
}

def get_gpu_memory_():
    """
    获取指定 GPU 的已用显存（单位：MB）
    """
    info = nvmlDeviceGetMemoryInfo(handle)
    used = info.used / 1024 ** 2  # 转换为 MB
    return used

def get_gpu_memory():
    try:
        used = get_gpu_memory_()
    except Exception as e:
        print(f"获取GPU显存信息时出错: {e}")
        used = 0
    return used

def get_cpu_memory():
    """
    获取指定 CPU 的已用显存（单位：MB）
    """
    # 获取当前进程的内存信息
    process = psutil.Process()
    # 模型加载前的 CPU 和 GPU 内存
    used = process.memory_info().rss / 1024 ** 2  # 转换为 MB
    return used

def local_chat(
    model_path: str | None = None,
    optimize_rule_path: str = None,
    gguf_path: str | None = None,
    max_new_tokens: int = 1024,
    cpu_infer: int = Config().cpu_infer,
    use_cuda_graph: bool = True,
    prompt_file : str | None = None,
    mode: str = "normal",
    force_think: bool = False,
):

    # 模型加载前的 CPU 和 GPU 内存
    cpu_before = get_cpu_memory()
    gpu_used_before = get_gpu_memory()

    torch.set_grad_enabled(False)

    Config().cpu_infer = cpu_infer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if mode == 'long_context':
        assert config.architectures[0] == "LlamaForCausalLM", "only LlamaForCausalLM support long_context mode"
        torch.set_default_dtype(torch.float16)
    else:
        torch.set_default_dtype(config.torch_dtype)

    with torch.device("meta"):
        if config.architectures[0] in custom_models:
            print("using custom modeling_xxx.py.")
            if (
                "Qwen2Moe" in config.architectures[0]
            ):  # Qwen2Moe must use flash_attention_2 to avoid overflow.
                config._attn_implementation = "flash_attention_2"
            if "Llama" in config.architectures[0]:
                config._attn_implementation = "eager"
            if "Mixtral" in config.architectures[0]:
                config._attn_implementation = "flash_attention_2"

            model = custom_models[config.architectures[0]](config)
        else:
            model = AutoModelForCausalLM.from_config(
                config, trust_remote_code=True, attn_implementation="flash_attention_2"
            )

    if optimize_rule_path is None:
        if config.architectures[0] in default_optimize_rules:
            print("using default_optimize_rule for", config.architectures[0])
            optimize_rule_path = default_optimize_rules[config.architectures[0]]
        else:
            optimize_rule_path = input(
                "please input the path of your rule file(yaml file containing optimize rules):"
            )

    if gguf_path is None:
        gguf_path = input(
            "please input the path of your gguf file(gguf file in the dir containing input gguf file must all belong to current model):"
        )
    optimize_and_load_gguf(model, optimize_rule_path, gguf_path, config)
    
    try:
            model.generation_config = GenerationConfig.from_pretrained(model_path)
    except:
            gen_config = GenerationConfig(
                max_length=128,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            model.generation_config = gen_config
    # model.generation_config = GenerationConfig.from_pretrained(model_path)
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
    model.eval()
    logging.basicConfig(level=logging.INFO)

    system = platform.system()
    if system == "Windows":
        os.system("cls")
    else:
        os.system("clear")

    # 模型加载后的 CPU 和 GPU 内存
    cpu_after_load = get_cpu_memory()
    gpu_used_after_load = get_gpu_memory()

    # 计算加载模型后的内存变化量
    cpu_change_load = cpu_after_load - cpu_before
    gpu_change_load = gpu_used_after_load - gpu_used_before

    print(f"加载模型后CPU内存变化量：{cpu_change_load:+.2f} MB")
    print(f"加载模型后GPU显存变化量：{gpu_change_load:+.2f} MB")
    print(f"\n")

    # 生成文本前的内存状态
    cpu_before_inference = get_cpu_memory()
    gpu_used_before_inference = get_gpu_memory()

    while True:
        content = input("Chat: ")
        if content.startswith('"""'):  # prefix """
            # multi lines input
            content = content[3:] + "\n"
            while True:
                line = input("")
                if line.endswith('"""'):
                    # end multi lines input
                    line = line[:-3]  # suffix """
                    if line:
                        content += line + "\n"
                    break
                else:
                    content += line + "\n"

        if content == "":
            if prompt_file != None:
                content = open(prompt_file, "r").read()
            else:
                content = "Please write a piece of quicksort code in C++."
        elif os.path.isfile(content):
            content = open(content, "r").read()
            
        messages = [{"role": "user", "content": content}]
        input_tensor = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        if force_think:
            token_thinks = torch.tensor([tokenizer.encode("<think>\\n",add_special_tokens=False)],device=input_tensor.device)
            input_tensor = torch.cat(
                [input_tensor, token_thinks], dim=1
            )
        if mode == 'long_context':
            assert Config().long_context_config['max_seq_len'] > input_tensor.shape[1] + max_new_tokens, \
            "please change max_seq_len in  ~/.ktransformers/config.yaml"
        
        if system != "Windows" and (config.architectures[0] == "DeepseekV2ForCausalLM" or "DeepseekV3ForCausalLM") and flashinfer_enabled:
            generated = prefill_and_generate(
                model, tokenizer, input_tensor.cuda(), max_new_tokens, use_cuda_graph, mode = mode, force_think = force_think,
                use_flashinfer_mla = True, num_heads = config.num_attention_heads, head_dim_ckv = config.kv_lora_rank, head_dim_kpe = config.qk_rope_head_dim, q_head_dim = config.qk_rope_head_dim + config.qk_nope_head_dim
            )
        else:
            generated = prefill_and_generate(
                model, tokenizer, input_tensor.cuda(), max_new_tokens, use_cuda_graph, mode = mode, force_think = force_think,
            )

        # 推理后的 CPU 和 GPU 内存
        cpu_after_inference = get_cpu_memory()
        gpu_used_after_inference = get_gpu_memory()

        # 计算推理后的内存变化量
        cpu_change_inference = cpu_after_inference - cpu_before_inference
        gpu_change_inference = gpu_used_after_inference - gpu_used_before_inference

        print(f"推理后CPU内存变化量：{cpu_change_inference:+.2f} MB")
        print(f"推理后GPU显存变化量：{gpu_change_inference:+.2f} MB")

if __name__ == "__main__":
    # 初始化 NVML
    nvmlInit()
    # 获取 GPU 句柄（假设使用的是 GPU 0）
    gpu_index = 0
    handle = nvmlDeviceGetHandleByIndex(gpu_index)
    
    # python -m ktransformers.local_chat --model_path deepseek-ai/DeepSeek-V2-Lite-Chat --gguf_path /root/autodl-tmp/models/DeepSeek-V2-Lite-Chat-GGUF
    # 将命令行参数直接传给函数
    fire.Fire(local_chat)

    # 关闭 NVML
    nvmlShutdown()