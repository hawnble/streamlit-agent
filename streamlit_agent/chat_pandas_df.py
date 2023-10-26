from langchain.agents import AgentType
from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
#from langchain.callbacks import StreamlitCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
#from langchain.schema import AIMessage, HumanMessage, SystemMessage

# from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
# from langchain.vectorstores import FAISS
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.prompts import FewShotPromptTemplate, PromptTemplate
# from langchain.llms import OpenAI

#from streamlit_chat import message
import streamlit as st
import pandas as pd
import os




# file_formats = {
#     "csv": pd.read_csv,
#     "xls": pd.read_excel,
#     "xlsx": pd.read_excel,
#     "xlsm": pd.read_excel,
#     "xlsb": pd.read_excel,
# }


from PIL import Image
im_logo = Image.open("로고.png")
im_symbol = Image.open("symbol.png")

# Submit 버튼 상태를 초기화하는 함수를 정의합니다.
def clear_submit():
    st.session_state["submit"] = False




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
def do_stuff_on_page_load():
    st.set_page_config(layout="wide", page_title="Pick-Chat! : Chat with DataFrame!", page_icon=im_symbol)#
do_stuff_on_page_load()
#st.set_page_config(layout="wide", page_title="Pick-Chat! : Chat with DataFrame!", page_icon=im_symbol)#
st.image(im_logo)
#st.title("Pick-Chat! : Chat with DataFrame!") #🦜 

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            #.st-emotion-cache-18ni7ap {visibility: hidden;}
            .st-emotion-cache-1pxazr7 {visibility: hidden;}
            .st-emotion-cache-eczf16 {visibility: hidden;}
            a.viewerBadge_container__r5tak.styles_viewerBadge__CvC9N {visibility: hidden;}
            </style>
            """
#st.markdown(hide_streamlit_style, unsafe_allow_html=True)

@st.cache_data
def load_data(url):
    df = pd.read_excel(url)
    return df
df = load_data("laptop_sdf_231026.xlsx")
#df = df.astype(str)
df.pop('Unnamed: 0')
#df

#예
examples = [
  {"input": "good_price", "output": "df_filtered = df[(df['Value_for_Money_Point'] >= df['Value_for_Money_Point'].quantile(0.75))]\ndf_sorted = df_filtered.groupby('Manufacturer').apply(lambda x: x.nlargest(1, 'Value_for_Money_Point')).reset_index(drop=True).sort_values(by='Price_won', ascending=True).head(5)"},
  {"input": "new&light", "output": "df_filtered = df[(df['CPU_Launch_Date'] >= 2023) & (df['inch_per_kg'] >= 13)]\ndf_sorted = df_filtered.groupby('Manufacturer').apply(lambda x: x.nlargest(1, 'Value_for_Money_Point')).reset_index(drop=True).sort_values(by='Price_won', ascending=True).head(5)"},
  {"input": "new&high_performance", "output": "df_filtered = df[(df['CPU_Launch_Date'] >= 2023) & (df['CPU_Score'] >= df['CPU_Score'].quantile(0.75)) & (df['GPU_Score'] >= df['GPU_Score'].quantile(0.75))]\ndf_sorted = df_filtered.groupby('Manufacturer').apply(lambda x: x.nlargest(1, 'Value_Point')).reset_index(drop=True).sort_values(by='Price_won', ascending=True).head(5)"},
  {"input": "new&good_display", "output": "df_filtered = df[(df['CPU_Launch_Date'] >= 2023) & (df['Display_Point'] >= df['Display_Point'].quantile(0.80))]\ndf_sorted = df_filtered.groupby('Manufacturer').apply(lambda x: x.nlargest(1, 'Value_for_Money_Point')).reset_index(drop=True).sort_values(by='Price_won', ascending=True).head(5)"},
  {"input": "new&good_service", "output": "df_filtered = df[(df['Manufacturer'].isin(['SAMSUNG', 'LG'])) & (df['CPU_Launch_Date'] >= 2023)]\ndf_sorted = df_filtered.sort_values(by=['Value_for_Money_Point', 'Price_won'], ascending=[False, True]).head(5)"},
  {"input": "light&high_performance", "output": "df_filtered = df[(df['inch_per_kg'] >= df['inch_per_kg'].quantile(0.74)) & (df['CPU_Score'] >= df['CPU_Score'].median()) & (df['GPU_Score'] >= df['GPU_Score'].median())]\ndf_sorted = df_filtered.groupby('Manufacturer').apply(lambda x: x.nlargest(1, 'Value_for_Money_Point')).reset_index(drop=True).sort_values(by='Price_won', ascending=True).head(5)"},
  {"input": "good_display&light", "output": "df_filtered = df[(df['Display_Point'] >= df['Display_Point'].quantile(0.85)) & (df['inch_per_kg'] >= df['inch_per_kg'].quantile(0.74))]\ndf_sorted = df_filtered.groupby('Manufacturer').apply(lambda x: x.nlargest(1, 'Value_for_Money_Point')).reset_index(drop=True).sort_values(by='Price_won', ascending=True).head(5)"},
  {"input": "light&good_service", "output": "df_filtered = df[df['Manufacturer'].isin(['SAMSUNG', 'LG']) & (df['inch_per_kg'] >= df['inch_per_kg'].quantile(0.74))]\ndf_sorted = df_sorted = df_filtered.groupby('Manufacturer').apply(lambda x: x.nlargest(3, 'Value_for_Money_Point')).reset_index(drop=True).sort_values(by='Price_won', ascending=True).head(5)"},
  {"input": "good_display&high_performance", "output": "df_filtered = df[(df['Display_Point'] >= df['Display_Point'].quantile(0.90)) & (df['CPU_Score'] >= df['CPU_Score'].quantile(0.90)) & (df['GPU_Score'] >= df['GPU_Score'].quantile(0.90))]\ndf_sorted = df_filtered.groupby('Manufacturer').apply(lambda x: x.nlargest(1, 'Value_Point')).reset_index(drop=True).sort_values(by='Price_won', ascending=True).head(5)"},
  {"input": "good_service&high_performance", "output": "df_filtered = df[df['Manufacturer'].isin(['SAMSUNG', 'LG']) & (df['CPU_Score'] >= 20000)]\ndf_sorted = df_filtered.sort_values(by=['Value_Point', 'Price_won'], ascending=[False, True]).head(5)"},
  {"input": "good_display&good_service", "output": "df_filtered = df[(df['Manufacturer'].isin(['SAMSUNG', 'LG'])) & (df['Display_Point'] >= df['Display_Point'].quantile(0.75))]\ndf_sorted = df_filtered.sort_values(by=['Value_for_Money_Point', 'Price_won'], ascending=[False, True]).head(5)"}
]

prefix_text = f'''너는 노트북을 전문적으로 추천해주는 챗봇 Pick-Chat!이야.
가격과 무게와 화면크기와 추천이유를 말해줘. 다른 정보는 요청시에만 제공해.
서로다른제조사로 제품을 최대 5개 추천하고 제품마다 줄바꿈을 해줘.
질문에 부합하는 데이터를 찾을 수 없는 경우에는 사용자에게 질문을 더 자세히 작성해달라고 요청해.
한글로 답변을 작성해. 하이퍼링크와 외부주소를 작성하면 안되. Display_Point, Value_for_Money_Point, Value_Point 는 공개하지마.
단 질문에 대한 데이터프레임에 적용하는 코드는 아래와 같이 작성해야해
{examples}
'''
   
# # SemanticSimilarityExampleSelector는 의미론적 의미에 따라 입력과 유사한 예제를 선택합니다.
# example_selector = SemanticSimilarityExampleSelector.from_examples(
#   examples,
#   OpenAIEmbeddings(openai_api_key=openai_api_key),  # 의미적 유사성을 측정하는 데 사용되는 임베딩을 생성하는 데 사용되는 임베딩 클래스입니다.
#   FAISS,  # 임베딩을 저장하고 유사성 검색을 수행하는 데 사용되는 VectorStore 클래스입니다.
#   k=1 # 생성할 예제 개수입니다.
# )

# similar_prompt = FewShotPromptTemplate(
#   example_selector=example_selector,  # 예제 선택에 도움이 되는 개체
#   example_prompt=example_prompt,  # 프롬프트
#   prefix=prefix_text,  # 프롬프트의 상단과 하단에 추가되는 사용자 지정 사항
#   suffix="Input: {noun}\nOutput:",
#   input_variables=["noun"],  # 프롬프트가 수신할 입력 항목
# )

# similar_prompt.format(noun=input('')


# OpenAI API 키 입력을 받습니다.
openai_api_key = st.secrets["openai_key"]

# 대화 기록을 초기화하거나 버튼을 눌러 대화 기록을 삭제합니다.
intro = f'{len(df)}개의 노트북이 있어요! 질문을 상세히 작성해 주시면 알맞는 제품을 찾아드릴게요!'
if "messages" not in st.session_state or st.button("Clear conversation history"):# 
    # 초기 대화 메시지 설정
    st.session_state["messages"] = [{"role": "assistant", "content": intro }]

# 이전 대화 내용을 표시합니다.
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 사용자 입력을 처리합니다.
if prompt := st.chat_input(placeholder="가볍고 빠른 노트북 추천해줄래? 무게는 1.5kg 이하면 좋아!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
     
    # OpenAI 모델 설정 및 실행
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    # ChatOpenAI 모델 초기화 및 설정
    llm = ChatOpenAI(
        temperature=0.25, model="gpt-4", openai_api_key=openai_api_key, streaming=True
    )

    # LangChain을 사용하여 pandas DataFrame 에이전트 생성 및 실행
    pandas_df_agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=False,
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
        #st.write(response)
        # st.markdown(response)


