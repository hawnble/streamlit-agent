from langchain.agents import AgentType
from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
#from langchain.callbacks import StreamlitCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
#from streamlit_chat import message
import torch
import streamlit as st
import pandas as pd
import os

df = pd.read_csv("laptop_sdf_231021.csv")
#df = df.astype(str)
df.pop('Unnamed: 0')
#df

from sentence_transformers import SentenceTransformer
q_df = pd.read_excel("EQC_df_231023.xlsx")

model = SentenceTransformer('jhgan/ko-sroberta-multitask')


file_formats = {
    "csv": pd.read_csv,
    "xls": pd.read_excel,
    "xlsx": pd.read_excel,
    "xlsm": pd.read_excel,
    "xlsb": pd.read_excel,
}




# Submit 버튼 상태를 초기화하는 함수를 정의합니다.
def clear_submit():
    st.session_state["submit"] = False

# 데이터를 로드하는 함수를 정의합니다. (캐싱 설정: 2시간)
# @st.cache_data(ttl="2h")
# def load_data(uploaded_file):
#     try:
#         # 파일 확장자 추출
#         ext = os.path.splitext(uploaded_file.name)[1][1:].lower()
#     except:
#         ext = uploaded_file.split(".")[-1]

#     # 파일 형식에 따라 적절한 함수로 데이터를 로드합니다.
#     if ext in file_formats:
#         return file_formats[ext](uploaded_file)
#     else:
#         # 지원하지 않는 파일 형식일 경우 에러 메시지 출력
#         st.error(f"Unsupported file format: {ext}")
#         return None
def cal_score(a, b):
    a = torch.tensor(a)  # NumPy 배열을 PyTorch 텐서로 변환
    b = torch.tensor(b)  # NumPy 배열을 PyTorch 텐서로 변환
    if len(a.shape) == 1: a = a.unsqueeze(0)
    if len(b.shape) == 1: b = b.unsqueeze(0)
    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]
    return torch.mm(a_norm, b_norm.transpose(0, 1)) * 100


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text=initial_text
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        # "/" is a marker to show difference
        # you don't need it
        self.text+=token
        self.container.markdown(self.text)

# Streamlit 페이지 설정
st.set_page_config(page_title="Pick-Chat! : Chat with DataFrame!", page_icon="🦜")
st.title("🦜 Pick-Chat! : Chat with DataFrame!")

# # 파일 업로드 위젯을 생성합니다.
# uploaded_file = st.file_uploader(
#     "Upload a Data file",
#     type=list(file_formats.keys()),
#     help="Various File formats are Support",
#     on_change=clear_submit,
# )

# # 파일이 업로드되지 않았을 때 경고 메시지를 표시합니다.
# if not uploaded_file:
#     st.warning(
#         "This app uses LangChain's `PythonAstREPLTool` which is vulnerable to arbitrary code execution. Please use caution in deploying and sharing this app."
#     )

# # 파일이 업로드된 경우 데이터를 로드합니다.
# if uploaded_file:
#     df = load_data(uploaded_file)

# OpenAI API 키 입력을 받습니다.
#openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
openai_api_key = st.secrets["openai_key"]
# 대화 기록을 초기화하거나 버튼을 눌러 대화 기록을 삭제합니다.
if "messages" not in st.session_state :#or st.sidebar.button("Clear conversation history")
    # 초기 대화 메시지 설정
    st.session_state["messages"] = [{"role": "assistant", "content": "질문을 상세히 작성해 주시면 정확한 답변이 가능해요!"}]

# 이전 대화 내용을 표시합니다.
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 사용자 입력을 처리합니다.
if prompt := st.chat_input(placeholder="가볍고 빠른 노트북 추천해줄래? 무게는 1.5kg 이하면 괜찮을거 같아!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    max_sim = -1
    max_idx = -1
    user_query = prompt
    for idx, val in q_df[['Embedded_Queries']].iterrows():
        embedded_user_query = model.encode(user_query)
        cos_sim = cal_score(embedded_user_query, q_df.loc[idx, 'Embedded_Queries'])
  
        if cos_sim > max_sim:
            max_sim = cos_sim.item()
            max_idx = idx

    similar_quary = q_df.loc[max_idx, 'Queries']
    code = q_df.loc[max_idx, 'codes']
    prefix_text = f'''너는 노트북을 전문적으로 추천해주는 챗봇 Pick-Chat!이야.
항상 가격과 무게와 화면크기와 장점을 말해줘. 다른 정보는 요청시에만 제공해.
서로다른제조사로 제품을 최대 5개 추천하고 제품마다 줄바꿈을 해줘.
질문에 부합하는 데이터를 찾을 수 없는 경우에는 사용자에게 질문을 더 자세히 작성해달라고 요청해.
항상 한글로 답변을 작성해. 절대 하이퍼링크와 외부주소를 작성하면 안되. Value_for_Money_Point 와 Value_Point 는 공개하지마.
단 질문에 대한 데이터프레임에 적용하는 코드는 아래와 같이 작성해야해.
질문: {similar_quary}
코드: {code}
'''

    # OpenAI 모델 설정 및 실행
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    # ChatOpenAI 모델 초기화 및 설정
    llm = ChatOpenAI(
        temperature=0.3, model="gpt-4-0613", openai_api_key=openai_api_key, streaming=True
    )

    # LangChain을 사용하여 pandas DataFrame 에이전트 생성 및 실행
    pandas_df_agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        handle_parsing_errors=True,
        prefix = prefix_text,
    )

    # Assistant 역할로 채팅 메시지를 표시합니다.
    with st.chat_message("assistant"):

        st.markdown("### Pick-Chat!")
        # here is the key, setup a empty container first
        chat_box=st.empty()
        stream_handler = StreamHandler(chat_box)
        # chat = ChatOpenAI(max_tokens=25, streaming=True, callbacks=[stream_handler])
        # st.markdown("### together box")

        # Streamlit 콜백 핸들러를 생성합니다.
        #st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

        # LangChain을 사용하여 대화를 진행하고 응답을 받습니다.
        response = pandas_df_agent.run(st.session_state.messages, callbacks=[stream_handler])

        # Assistant의 응답을 대화 기록에 추가하고 출력합니다.
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(prefix_text)
        # st.markdown(response)
