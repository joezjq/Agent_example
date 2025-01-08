import json
import os
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage,SystemMessage
import config

#配置环境
os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY
openai_api_key = os.getenv("OPENAI_API_KEY")

max_interactions = 5

def qa_generate(user_prompt, assistant_prompt):

    #初始化两个LLM: USER 和 ASSISTANT
    user_llm = ChatOpenAI(model = "deepseek-chat", openai_api_key = openai_api_key, temperature = 0.7, base_url = "https://api.deepseek.com")
    assistant_llm = ChatOpenAI(model = "deepseek-chat", openai_api_key = openai_api_key, temperature = 0.7, base_url = "https://api.deepseek.com")

    user_context = [SystemMessage(content = "你是学生: "+ user_prompt)]
    assistant_context = [SystemMessage(content = "你是老师："+ assistant_prompt)]

    qa_trace = []

    for _ in range(max_interactions):
        #User LLM说话
        user_context.append(SystemMessage(content="注意，你是学生，请勿作为老师发言"))
        user_response = user_llm(messages = user_context)
        print(f'User:{user_response.content}')
        qa_trace.append("User: " + user_response.content)
        user_context.pop()

        #Assistant说话
        assistant_context.append(HumanMessage(content=user_response.content))#添加User回复到上下文
        assistant_context.append(SystemMessage(content="注意，你是老师，只需要回答学生的问题，不要作为学生提问"))
        assistant_response = assistant_llm(messages = assistant_context)
        print(f'Assistant:'+assistant_response.content)
        qa_trace.append("Assistant:" + assistant_response.content)
        assistant_context.pop()
        user_context.append(AIMessage(content=assistant_response.content))
    
    return qa_trace

if __name__ == '__main__':
    user_prompt = config.user_prompt
    assistant_prompt = config.assistant_prompt
    results_dict = {}
    
    for i in range(1,10):
        print(f"--------------------------------第{i}组对话---------------------------------------")
        result = qa_generate(user_prompt,assistant_prompt)
        results_dict[f"exp{i}"] = result

    with open("result/results_dict.json", "w", encoding="utf-8") as file:
        json.dump(results_dict, file, ensure_ascii=False, indent=3)  # ensure_ascii=False是保留中文,indent=4是缩进


    