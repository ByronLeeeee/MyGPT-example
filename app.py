import gradio as gr
from MyGPT import ask_question, ask_ChatGLM
from MyGPT_langchain import docx2txt, lawEmbedding, LawQA


with gr.Blocks(title="MyGPT") as app:  # 设置一个名为MyGPT的基本块
    # 设置三个标签页，并分别命名
    with gr.Tab("导入数据库"):
        def TextEmbedding(path):
            result_D2T = docx2txt(path)
            result_Emb = lawEmbedding(path)
            return result_Emb

        with gr.Column():  # 新建一列
            with gr.Row():  # 新建一行，且一行中有两个组件
                # 新建一个文本栏，用于输入文件夹路径，且给出默认值'./laws'
                path = gr.Textbox(
                    value='./laws', label='请输入文件夹路径', lines=1, interactive=True)
                # 新建一个按钮，用于让程序开始运行
                button = gr.Button(value='开始导入', variant='primary')
            with gr.Row():  # 再次新建一列
                log = gr.Textbox(label='状态', value='等待导入中……')  # 用于获取状态

        button.click(fn=TextEmbedding, inputs=path, outputs=log)

    with gr.Tab("问答"):
        # 创建一个选项框，用于选择聊天模式
        mode = gr.Radio(choices=['资料库模式', '普通聊天'], label='聊天模式')
        # 创建一个聊天机器人组件，作为我们的主要聊天界面
        chatbot = gr.Chatbot(value=[[None, '您好！请问有什么能帮到您？']], label='AI智能助理')
        msg = gr.Textbox(label='对话框')  # 创建一个用于输入对话的文本框
        clear = gr.Button("清空对话")  # 用于清空全部历史记录
        msg_cache = gr.TextArea(visible=False)  # 用于暂存我们的问题

        # 为了更像一个聊天工具，我们制作这个函数；
        # 程序会在接收到问题后，马上清空对话框的内容，并在聊天界面中显示我们的问题
        def user(user_message, history):
            return '', history + [[user_message, None]], user_message

        # 核心聊天函数
        def respond(message, chat_history, mode):
            if mode == '资料库模式':  # 通过选择的聊天模式决定从哪个函数获取回答
                bot_message = LawQA(message)
            if mode == '普通聊天':
                # 根据之前的功能编写，我们需要传入一个符合ChatGPT要求的含有身份字典的数组，以及具体的聊天模型
                bot_message = ask_question(
                    [{"role": "user", "content": message}], model_engine="gpt-3.5-turbo")
            if mode == '':
                bot_message = "抱歉，请先选择聊天模式。"
            chat_history.append([None, bot_message])
            return chat_history
        # 点击后就会先执行user函数，记录我们的问题，然后再执行respond函数获取回答
        msg.submit(user, [msg, chatbot], [msg, chatbot, msg_cache]).then(
            respond, [msg_cache, chatbot, mode], chatbot)
        clear.click(lambda: None, None, chatbot, queue=False)  # 一键清空全部记录

    with gr.Tab("ChatGLM问答"):
        #创建一个聊天机器人组件，作为我们的主要聊天界面
        chatbot = gr.Chatbot(value=[[None,'您好！请问有什么能帮到您？']],label='ChatGLM AI助手')
        msg = gr.Textbox(label='对话框') #创建一个用于输入对话的文本框
        clear = gr.Button("清空对话") #用于清空全部历史记录
        msg_cache = gr.TextArea(visible=False) #用于暂存我们的问题

        #为了更像一个聊天工具，程序会在接收到问题后，马上清空对话框的内容，并在聊天界面中显示我们的问题
        def user(user_message, history):
            return '',history + [[user_message, None]],user_message

        #核心聊天函数
        def respond(message, chat_history):
            # bot_message = ask_ChatGLM(message,)
            # chat_history.append([None,bot_message])
            return chat_history
        #点击后就会先执行user函数，记录我们的问题，然后再执行respond函数获取回答
        msg.submit(user,[msg,chatbot],[msg,chatbot,msg_cache]).then(respond, [msg_cache, chatbot], chatbot)
        clear.click(lambda: None, None, chatbot, queue=False) #一键清空全部记录

    with gr.Tab('常设功能'):
        import json

        def read_json(file_path='./prompt.json'):
            with open(file_path, 'r', encoding="utf-8") as file:
                data = json.load(file)  # 读取 JSON 文件内容

            result = []  # 保存键值对的数组
            for key, value in data.items():  # 遍历键值对
                result.append([key, value])  # 将键值对添加到数组中，以列表形式保存

            return result

        def getAnswer(system, user):
            if not system or not user:

                # 如果系统或用户输入为空，则返回默认提示
                answer = '请输入身份模版和您的提问'
            else:

                # 构建包含系统和用户输入的提示列表
                prompt = [{"role": "system", "content": system},
                          {"role": "user", "content": user}]
                
                # 使用ask_question函数根据提示列表和指定的模型引擎进行提问，并获取回答
                answer = ask_question(prompt, model_engine="gpt-3.5-turbo")
                
            return answer

        promptList = read_json('./prompt.json')  # 读取我们的Prompt json文件，并取出里面的值

        with gr.Row():
            usage = gr.TextArea(label='功能', lines=2)
            systemPrompt = gr.TextArea(label='身份模版', lines=2)
        userPrompt = gr.TextArea(label='提问', lines=5)
        message = gr.TextArea(label='ChatGPT', lines=20)
        btn = gr.Button("提交", variant='primary')
        btn.click(getAnswer, [systemPrompt, userPrompt], message)
        gr.Markdown("## 身份模版参考（点击可用）")
        examples = gr.Examples(promptList, [usage, systemPrompt])

    # with gr.Tab("新增功能示例"):
    #     #from xxx import xxx 导入相关的功能
    #     def newQA(content,question):
    #         '''
    #         实现功能的代码，可以根据官方示例制作
    #         '''
    #         return answer
    #     content = gr.TextArea("读取内容")
    #     question = gr.TextArea("问题")
    #     answer = gr.TextArea('回答')
    #     btn = gr.Button('提交')
    #     btn.click(newQA,[content,question],answer)


# 启动APP
app.launch()
