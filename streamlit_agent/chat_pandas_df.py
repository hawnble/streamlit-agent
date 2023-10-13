from langchain.agents import AgentType
from langchain.agents import create_pandas_dataframe_agent
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
import streamlit as st
import pandas as pd
import os

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
@st.cache_data(ttl="2h")
def load_data(uploaded_file):
    try:
        # 파일 확장자 추출
        ext = os.path.splitext(uploaded_file.name)[1][1:].lower()
    except:
        ext = uploaded_file.split(".")[-1]

    # 파일 형식에 따라 적절한 함수로 데이터를 로드합니다.
    if ext in file_formats:
        return file_formats[ext](uploaded_file)
    else:
        # 지원하지 않는 파일 형식일 경우 에러 메시지 출력
        st.error(f"Unsupported file format: {ext}")
        return None

# Streamlit 페이지 설정
st.set_page_config(page_title="LangChain: Chat with pandas DataFrame", page_icon="🦜")
st.title("🦜 LangChain: Chat with pandas DataFrame")

# 파일 업로드 위젯을 생성합니다.
uploaded_file = st.file_uploader(
    "Upload a Data file",
    type=list(file_formats.keys()),
    help="Various File formats are Support",
    on_change=clear_submit,
)

# 파일이 업로드되지 않았을 때 경고 메시지를 표시합니다.
if not uploaded_file:
    st.warning(
        "This app uses LangChain's `PythonAstREPLTool` which is vulnerable to arbitrary code execution. Please use caution in deploying and sharing this app."
    )

# 파일이 업로드된 경우 데이터를 로드합니다.
if uploaded_file:
    df = load_data(uploaded_file)

# OpenAI API 키 입력을 받습니다.
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

# 대화 기록을 초기화하거나 버튼을 눌러 대화 기록을 삭제합니다.
if "messages" not in st.session_state or st.sidebar.button("Clear conversation history"):
    # 초기 대화 메시지 설정
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# 이전 대화 내용을 표시합니다.
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 사용자 입력을 처리합니다.
if prompt := st.chat_input(placeholder="What is this data about?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # OpenAI 모델 설정 및 실행
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    # ChatOpenAI 모델 초기화 및 설정
    llm = ChatOpenAI(
        temperature=0, model="gpt-4-0613", openai_api_key=openai_api_key, streaming=True
    )

    # LangChain을 사용하여 pandas DataFrame 에이전트 생성 및 실행
    pandas_df_agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        handle_parsing_errors=True,
    )

    # Assistant 역할로 채팅 메시지를 표시합니다.
    with st.chat_message("assistant"):
        # Streamlit 콜백 핸들러를 생성합니다.
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        
        # LangChain을 사용하여 대화를 진행하고 응답을 받습니다.
        response = pandas_df_agent.run(st.session_state.messages, callbacks=[st_cb])
        
        # Assistant의 응답을 대화 기록에 추가하고 출력합니다.
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
