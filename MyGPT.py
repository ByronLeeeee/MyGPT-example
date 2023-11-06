import os
import openai
from transformers import AutoTokenizer, AutoModel
import torch


#如需设置网络代理，请删掉前方的#号并设置具体的代理信息，如果没有网络问题可以不用修改；
# os.environ["http_proxy"] = "http://127.0.0.1:xxxx"
# os.environ["https_proxy"] = "http://127.0.0.1:xxxx"

openai.api_key = "YOUR-API-KEYS" #设置申领到的OpenAI API Key，也可以使用os.environ["OPENAI_API_KEY"]的办法
model_engine = "gpt-3.5-turbo" # 或者其它你申请到的模型ID

def ask_question(history, model_engine="gpt-3.5-turbo"):
    """
    用于向ChatGPT提问的函数

    参数:
    history (str): 历史对话记录，用于生成回答的输入
    model_engine (str): ChatGPT模型引擎的ID，默认为"gpt-3.5-turbo"

    返回:
    str: ChatGPT回答中的文本部分，去除多余空格后的结果
    """
    # 调用OpenAI API，获取ChatGPT的回答
    completions = openai.ChatCompletion.create(
        model=model_engine,  # ChatGPT模型引擎ID
        messages=history,  # 传入history
        max_tokens=1024,  # 最大生成字符数量
        n=1,  # 返回结果数量
        stop=None,  # 结束生成条件
        temperature=0.7  # 生成多样性控制参数, 0.1最严谨，1最具有创造性
    )

    message = completions.choices[0].message.content  # 获取ChatGPT回答中的文本部分
    return message.strip()  # 去除多余空格并返回结果

def ask_ChatGLM(prompt):
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
    # 使用给定的prompt参数加载预训练的tokenizer
    # tokenizer用于将输入的文本转换为模型可以理解的格式

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    # 检查系统是否支持CUDA加速。如果支持，则将device设置为"cuda"，否则检查是否支持多处理器（mps）加速，如果支持则将device设置为"mps"，否则将device设置为"cpu"

    model = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True, device=device)
    # 使用给定的device参数加载预训练的模型
    # model用于生成聊天回复

    model = model.eval()
    # 将模型设置为评估模式，以便进行预测而不是训练

    response = model.chat(tokenizer, prompt)
    # 使用model和tokenizer将给定的prompt转换为模型可以理解的格式，并生成聊天回复

    return response
    # 返回生成的聊天回复作为函数的结果


if __name__ == '__main__':
    #创建一个名为history的列表
    history = []
    print("请输入您的问题（或输入'退出'结束）：")
    # 主循环
    while True:
        # 获取用户输入的问题
        prompt = input("问题：")
        #把用户的问题作为user内容添加到history中
        history.append({"role":"user","content":prompt})
        # 检查是否要退出程序
        if prompt.lower() == "退出":
            break
        # 调用函数获取回答
        response = ask_question(history, model_engine)
        #把ChatGPT的回答作为assistant内容添加到history中
        history.append({"role":"assistant","content":response})
        # 输出回答
        print("回答：", response)