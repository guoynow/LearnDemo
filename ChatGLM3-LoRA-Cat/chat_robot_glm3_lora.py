# 导入操作系统相关模块，用于获取环境变量等操作
import os
# 导入PyTorch库，用于深度学习相关操作
import torch
# 导入线程模块，用于实现多线程操作
from threading import Thread
# 导入类型注解模块，用于定义联合类型
from typing import Union
# 导入路径操作模块，用于处理文件路径
from pathlib import Path
# 从peft库导入自动加载PEFT模型和PEFT模型类
from peft import AutoPeftModelForCausalLM, PeftModelForCausalLM
# 从transformers库导入各种模型、分词器和生成相关的类
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer
)
# 导入Tornado Web框架相关模块，用于构建Web服务
import tornado.web
import tornado.ioloop
from tornado.web import RequestHandler

# 定义模型类型，支持预训练模型和PEFT模型
ModelType = Union[PreTrainedModel, PeftModelForCausalLM]
# 定义分词器类型，支持预训练分词器和快速预训练分词器
TokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

# 从环境变量中获取模型路径，如果未设置则使用默认路径
MODEL_PATH = os.environ.get('MODEL_PATH', '/home/guoyuanow/code/ChatGLM3/finetune/output/checkpoint-500')

# 定义加载模型和分词器的函数
# Union[str, Path] 表示该参数可以接受两种类型的输入
def load_model_and_tokenizer(
        model_dir: Union[str, Path], trust_remote_code: bool = True
) -> tuple[ModelType, TokenizerType]:
    # 将模型路径转换为Path对象并解析绝对路径
    model_dir = Path(model_dir).expanduser().resolve()
    # 检查是否存在适配器配置文件，如果存在则加载PEFT模型
    if (model_dir / 'adapter_config.json').exists():
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=trust_remote_code, device_map='auto')
        tokenizer_dir = model.peft_config['default'].base_model_name_or_path
    else:
        # 若不存在适配器配置文件，则加载普通的预训练模型
        model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=trust_remote_code, device_map='auto')
        tokenizer_dir = model_dir

    # 从指定路径加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir,  # 分词器的预训练模型路径
        trust_remote_code=trust_remote_code,  # 是否信任远程代码，用于加载自定义的分词器代码
        encode_special_tokens=True,  # 是否对特殊标记（如 <bos>, <eos> 等）进行编码
        use_fast=False  # 是否使用快速版本的分词器
    )
    return model, tokenizer

# 调用函数加载模型和分词器
model, tokenizer = load_model_and_tokenizer(MODEL_PATH, trust_remote_code=True)

# 定义停止条件类，用于在生成文本时判断是否停止
# 该类继承自Hugging Face Transformers库中的 StoppingCriteria 基类，
# 用于自定义文本生成的停止条件 —— 当模型生成特定token序列时，触发停止机制，
# 避免生成过长或不必要的内容。
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # 获取模型配置中的结束标记ID
        stop_ids = model.config.eos_token_id
        # 遍历结束标记ID，如果当前生成的最后一个标记是结束标记，则返回True

        # 如果stop_ids不是列表，将其转换为列表
        if not isinstance(stop_ids, list):
            stop_ids = [stop_ids]

        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

# 定义聊天机器人API函数，处理用户输入并生成响应
def chatbot_api(infos):
    # 初始化对话历史列表
    history = []
    # 设置生成文本的最大长度
    max_length = 8192
    # 设置采样概率阈值
    top_p = 0.7
    # 设置温度参数，控制生成文本的随机性
    temperature = 0.9
    # 创建停止条件对象
    stop = StopOnTokens()

    print("Welcome to the ChatGLM3-6B CLI chat. Type your messages below.")

    # 获取用户输入
    user_input = infos
    
    # if user_input.lower() in ["exit", "quit"]:
    #     break

    # 将用户输入添加到对话历史中
    history.append([user_input, ""])

    # 构建对话消息列表
    messages = []
    for idx, (user_msg, model_msg) in enumerate(history):
        if idx == len(history) - 1 and not model_msg:
            messages.append({"role": "user", "content": user_msg})
            break
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if model_msg:
            messages.append({"role": "assistant", "content": model_msg})
    # 使用分词器将对话消息转换为模型输入张量
    # 使用分词器的聊天模板方法处理对话消息，将其转换为模型可接受的输入格式
    model_inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True, # 在末尾添加生成提示标记，告诉模型开始生成回复
        tokenize=True, # 将文本转换为token ID序列
        return_tensors="pt"
    ).to(model.device)
    # 创建文本流生成器
    streamer = TextIteratorStreamer(
        tokenizer=tokenizer,
        timeout=60,
        skip_prompt=True, # 跳过输入提示部分，只输出生成的新内容
        skip_special_tokens=True # 跳过特殊token
    )
    # 设置模型生成参数
    generate_kwargs = {
        "input_ids": model_inputs,  # 模型的输入张量，由分词器处理对话消息后得到
        "streamer": streamer,  # 文本流生成器，用于流式输出生成的文本
        "max_new_tokens": max_length,  
        "do_sample": True,  # do_sample=False会使用贪心解码，总是选择最高概率的token，结果更确定但缺乏变化
        "top_p": top_p,  
        "temperature": temperature,  
        "stopping_criteria": StoppingCriteriaList([stop]),  # 停止条件列表，包含自定义的停止条件
        "repetition_penalty": 1.2,  # 重复惩罚系数，用于减少生成文本中的重复内容
        "eos_token_id": model.config.eos_token_id, 
    }

    # 创建并启动一个线程来执行模型生成任务
    # target: 指定线程要执行的目标函数，这里为模型的generate方法，用于生成文本
    # kwargs: 传递给目标函数的关键字参数，这里包含模型生成所需的各种参数
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    print("ChatGLM3:", end="", flush=True) # flush=True: 立即刷新输出缓冲区，确保提示立即显示而不等待缓冲区满
    # 从流生成器中获取新生成的token并打印

    for new_token in streamer:
        if new_token:
            print(new_token, end="", flush=True)
            history[-1][1] += new_token

    # 去除生成响应的首尾空白字符
    history[-1][1] = history[-1][1].strip()
    return history[-1][1]

# 代码不用看
# 定义基础请求处理类，用于解决JS跨域请求问题
class BaseHandler(RequestHandler):
    """解决JS跨域请求问题"""

    def set_default_headers(self):
        self.set_header('Access-Control-Allow-Origin', '*')
        self.set_header('Access-Control-Allow-Methods', 'POST, GET')
        self.set_header('Access-Control-Max-Age', 1000)
        self.set_header('Access-Control-Allow-Headers', '*')
        # self.set_header('Content-type', 'application/json')

# 定义索引请求处理类，处理GET请求
class IndexHandler(BaseHandler):
    # 添加一个处理get请求方式的方法
    def get(self):
        # 向响应中，添加数据
        infos = self.get_query_argument("infos")
        print("\nQ:", infos)
        # 捕捉服务器异常信息
        try:
            result = chatbot_api(infos=infos)#调用训练好的模型，预测得到结果，结果给了result
        except Exception as e:
            print(e)
            result = "服务器内部错误"
        # print("A:", "".join(result))
        self.write("".join(result)) #发送给前端

if __name__ == '__main__':
    # 创建一个Tornado Web应用对象
    # 创建一个Tornado Web应用对象，通过路由配置将URL路径 `/api/chatbot` 映射到 `IndexHandler` 类
    # 当客户端请求 `/api/chatbot` 路径时，会由 `IndexHandler` 类处理该请求
    app = tornado.web.Application([(r'/api/chatbot', IndexHandler)])
    # 绑定一个监听端口
    app.listen(6006)
    # 启动web程序，开始监听端口的连接
    tornado.ioloop.IOLoop.current().start()
