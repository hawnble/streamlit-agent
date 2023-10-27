from langchain.agents import AgentType
from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

import streamlit as st
import pandas as pd
import os

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

# col1, col2 = st.columns([3, 1])
# with col1:
#st.set_page_config(layout="wide", page_title="Pick-Chat! : Chat with DataFrame!", page_icon=im_symbol)#
st.image(im_logo)
#st.title("Pick-Chat! : Chat with DataFrame!") #🦜 

# Streamlit 기본 테두리 제거(잘안됨)
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

def filter_and_select_laptops(df, first_query, num_items=3):
    # 데이터프레임 복사
    df_filtered = df.copy()  # 초기 데이터프레임으로 시작
    condition = None

    # 첫 번째 쿼리에 따라 조건 설정
    if '빠르' in first_query or '빠른' in first_query or '빨랐' in first_query or '빠릿' in first_query or '쾌적' in first_query or '속도' in first_query or 'fast' in first_query:
        if condition is None:
            condition = fast
        else:
            condition = condition & fast
    if '최신' in first_query or '신형' in first_query or '요즘' in first_query or '최근' in first_query or '새로' in first_query or 'new' in first_query:
        if condition is None:
            condition = new
        else:
            condition = condition & new
    if '가벼' in first_query or '가볍' in first_query or '안무거' in first_query or '무겁지' in first_query or '들고' in first_query or 'light' in first_query:
        if condition is None:
            condition = light
        else:
            condition = condition & light
    if '성능' in first_query or '고성능' in first_query or '게임' in first_query or '게이밍' in first_query or 'performance' in first_query or 'game' in first_query or 'gaming' in first_query or '그래픽' in first_query or 'graphic' in first_query:
        if condition is None:
            condition = performance
        else:
            condition = condition & performance
    if '화질' in first_query or '밝' in first_query or '선명' in first_query or '주사율' in first_query or 'ppi' in first_query or 'PPI' in first_query or 'display' in first_query or 'Display' in first_query or '디스플' in first_query or 'hz' in first_query or 'HZ' in first_query or 'Hz' in first_query:
        if condition is None:
            condition = display
        else:
            condition = condition & display
    if 'AS' in first_query or 'as' in first_query or 'Service' in first_query or 'service' in first_query or '에이에' in first_query or '국산' in first_query or '국내' in first_query or '삼성' in first_query or '엘지' in first_query or 'LG' in first_query or 'lg' in first_query:
        if condition is None:
            condition = AS
        else:
            condition = condition & AS
    if '큰' in first_query or '크고' in first_query or '크면' in first_query:
        if condition is None:
            condition = large
        else:
            condition = condition & large
    if 'pd' in first_query or 'PD' in first_query or '피디' in first_query or '충전' in first_query:
        if condition is None:
            condition = pd_charge

    # 조건에 따라 데이터프레임 필터링
    if condition is not None:
        df_filtered = df_filtered[condition]

    # 제조사별로 1개씩 선택
    selected_items = []
    for manufacturer in df_filtered['Manufacturer'].unique():
        manufacturer_df = df_filtered[df_filtered['Manufacturer'] == manufacturer]
        if not manufacturer_df.empty:
            selected_item = manufacturer_df.nlargest(1, 'Value_for_Money_Point')
            selected_items.append(selected_item)

    # num_items 개수에 도달할 때까지 추가적인 항목 선택
    while len(selected_items) < num_items:
        remaining_df = df_filtered[~df_filtered.index.isin([item.index[0] for item in selected_items])]
        if remaining_df.empty:
            break
        selected_item = remaining_df.nlargest(1, 'Value_for_Money_Point')
        if not selected_item.empty:
            selected_items.append(selected_item)

    # 선택된 항목들을 하나의 데이터프레임으로 결합하고 가격에 따라 정렬
    if selected_items:
        df_sorted = pd.concat(selected_items).sort_values(by='Price_won', ascending=True).head(num_items)
    else:
        df_sorted = pd.DataFrame()

    return df_sorted.reset_index(drop=True)

fast = (df['CPU_Score'] >= 20000)
new = (df['CPU_Launch_Date'] >= 2023.01)
light = (df['inch_per_kg'] >= 9)
performance = (df['CPU_Score'] >= df['CPU_Score'].quantile(0.74)) & (df['GPU_Score'] >= df['GPU_Score'].quantile(0.74))
display = (df['Display_Point'] >= df['Display_Point'].quantile(0.74))
AS = df['Manufacturer'].isin(['SAMSUNG', 'LG'])
large = (df['inch'] >= 15)
pd_charge = (df['PD충전'] == 'USB-PD')

#예
examples = [
  {"input": "good_price", "output": "df_filtered = df[(df['Value_for_Money_Point'] >= df['Value_for_Money_Point'].quantile(0.75))]\ndf_sorted = df_filtered.groupby('Manufacturer').apply(lambda x: x.nlargest(1, 'Value_for_Money_Point')).reset_index(drop=True).sort_values(by='Price_won', ascending=True).head(3)"},
  {"input": "new&light", "output": "df_filtered = df[(df['CPU_Launch_Date'] >= 2023) & (df['inch_per_kg'] >= 13)]\ndf_sorted = df_filtered.groupby('Manufacturer').apply(lambda x: x.nlargest(1, 'Value_for_Money_Point')).reset_index(drop=True).sort_values(by='Price_won', ascending=True).head(3)"},
  {"input": "new&high_performance", "output": "df_filtered = df[(df['CPU_Launch_Date'] >= 2023) & (df['CPU_Score'] >= df['CPU_Score'].quantile(0.75)) & (df['GPU_Score'] >= df['GPU_Score'].quantile(0.75))]\ndf_sorted = df_filtered.groupby('Manufacturer').apply(lambda x: x.nlargest(1, 'Value_Point')).reset_index(drop=True).sort_values(by='Price_won', ascending=True).head(3)"},
  {"input": "new&good_display", "output": "df_filtered = df[(df['CPU_Launch_Date'] >= 2023) & (df['Display_Point'] >= df['Display_Point'].quantile(0.80))]\ndf_sorted = df_filtered.groupby('Manufacturer').apply(lambda x: x.nlargest(1, 'Value_for_Money_Point')).reset_index(drop=True).sort_values(by='Price_won', ascending=True).head(3)"},
  {"input": "new&good_service", "output": "df_filtered = df[(df['Manufacturer'].isin(['SAMSUNG', 'LG'])) & (df['CPU_Launch_Date'] >= 2023)]\ndf_sorted = df_filtered.sort_values(by=['Value_for_Money_Point', 'Price_won'], ascending=[False, True]).head(3)"},
  {"input": "light&high_performance", "output": "df_filtered = df[(df['inch_per_kg'] >= df['inch_per_kg'].quantile(0.74)) & (df['CPU_Score'] >= df['CPU_Score'].median()) & (df['GPU_Score'] >= df['GPU_Score'].median())]\ndf_sorted = df_filtered.groupby('Manufacturer').apply(lambda x: x.nlargest(1, 'Value_for_Money_Point')).reset_index(drop=True).sort_values(by='Price_won', ascending=True).head(3)"},
  {"input": "good_display&light", "output": "df_filtered = df[(df['Display_Point'] >= df['Display_Point'].quantile(0.85)) & (df['inch_per_kg'] >= df['inch_per_kg'].quantile(0.74))]\ndf_sorted = df_filtered.groupby('Manufacturer').apply(lambda x: x.nlargest(1, 'Value_for_Money_Point')).reset_index(drop=True).sort_values(by='Price_won', ascending=True).head(3)"},
  {"input": "light&good_service", "output": "df_filtered = df[df['Manufacturer'].isin(['SAMSUNG', 'LG']) & (df['inch_per_kg'] >= df['inch_per_kg'].quantile(0.74))]\ndf_sorted = df_sorted = df_filtered.groupby('Manufacturer').apply(lambda x: x.nlargest(3, 'Value_for_Money_Point')).reset_index(drop=True).sort_values(by='Price_won', ascending=True).head(3)"},
  {"input": "good_display&high_performance", "output": "df_filtered = df[(df['Display_Point'] >= df['Display_Point'].quantile(0.90)) & (df['CPU_Score'] >= df['CPU_Score'].quantile(0.90)) & (df['GPU_Score'] >= df['GPU_Score'].quantile(0.90))]\ndf_sorted = df_filtered.groupby('Manufacturer').apply(lambda x: x.nlargest(1, 'Value_Point')).reset_index(drop=True).sort_values(by='Price_won', ascending=True).head(3)"},
  {"input": "good_service&high_performance", "output": "df_filtered = df[df['Manufacturer'].isin(['SAMSUNG', 'LG']) & (df['CPU_Score'] >= 20000)]\ndf_sorted = df_filtered.sort_values(by=['Value_Point', 'Price_won'], ascending=[False, True]).head(3)"},
  {"input": "good_display&good_service", "output": "df_filtered = df[(df['Manufacturer'].isin(['SAMSUNG', 'LG'])) & (df['Display_Point'] >= df['Display_Point'].quantile(0.75))]\ndf_sorted = df_filtered.sort_values(by=['Value_for_Money_Point', 'Price_won'], ascending=[False, True]).reset_index(drop=True).head(3)"}
]
prefix_text = f'''너는 노트북을 전문적으로 추천해주는 챗봇 Pick-Chat!이야.
가격과 무게와 화면크기와 추천이유를 말해줘.
서로다른제조사로 제품을 최대 3개 추천하고 제품마다 줄바꿈을 해줘.
질문에 부합하는 데이터를 찾을 수 없는 경우에는 사용자에게 질문을 더 자세히 작성해달라고 요청해.
한글로 답변을 작성해. 하이퍼링크와 외부주소를 작성하면 안되. Display_Point, Value_for_Money_Point, Value_Point 는 공개하지마.
단 질문에 대한 데이터프레임에 적용하는 코드는 아래와 같이 작성해야해
{examples}
'''
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

    df_s = filter_and_select_laptops(df, prompt, num_items=3).head()
    if len(df_s) >= 1 and len(st.session_state.messages) == 2:
        # OpenAI 모델 설정 및 실행
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
    
        # ChatOpenAI 모델 초기화 및 설정
        llm_t = ChatOpenAI(
            temperature=0.24, model="gpt-3.5-turbo", openai_api_key=openai_api_key, streaming=True
        )
    
        # LangChain을 사용하여 pandas DataFrame 에이전트 생성 및 실행
        pandas_df_agent = create_pandas_dataframe_agent(
            llm_t,
            df_s,
            verbose=False,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            handle_parsing_errors=True,
            prefix = '''너는 dataframe의 노트북을 설명해주는 챗봇 Pick-Chat!이야. 고민없이 바로 설명해.
제품마다 가격과 무게와 화면크기와 추천이유를 꼭 말하고 줄바꿈을 해줘.
반드시 한글로 작성해. '''
        )
        
        messages = [
        SystemMessage(
        content=f'''너는 dataframe {df_s}의 모든 노트북을 설명해주는 챗봇 Pick-Chat!이야.
제품마다 가격과 무게와 화면크기와 추천이유를 꼭 말하고 줄바꿈을 해줘.
반드시 한글로 작성해. '''
        ),
        HumanMessage(
        content=prompt
        ),
        ]
                
        # Assistant 역할로 채팅 메시지를 표시합니다.
        with st.chat_message("assistant"):
    
            st.markdown("### Pick-Chat!")
            # here is the key, setup a empty container first
            chat_box=st.empty()
            stream_handler = StreamHandler(chat_box)

            st.image(f'output_images/{df_s.loc[0, "No"]}.png', width = 200);
            #st.image(f'output_images/{df_s.loc[1, "No"]}.png', width = 200);
            #st.image(f'output_images/{df_s.loc[2, "No"]}.png', width = 200)
            
            # LangChain을 사용하여 대화를 진행하고 응답을 받습니다.
            #response = pandas_df_agent.run(st.session_state.messages, callbacks=[stream_handler])
            response = llm_t(messages, callbacks=[stream_handler])
            # Assistant의 응답을 대화 기록에 추가하고 출력합니다.
            st.session_state.messages.append({"role": "assistant", "content": response})
            #st.write(response)
            # st.markdown(response)

    elif len(df_s) >= 1 and len(st.session_state.messages) > 2:
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
            df_s,
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
           
            # LangChain을 사용하여 대화를 진행하고 응답을 받습니다.
            response = pandas_df_agent.run(st.session_state.messages, callbacks=[stream_handler])
    
            # Assistant의 응답을 대화 기록에 추가하고 출력합니다.
            st.session_state.messages.append({"role": "assistant", "content": response})
            #st.write(response)
            # st.markdown(response)
    
    else:
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
            df, df_s,
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

            # LangChain을 사용하여 대화를 진행하고 응답을 받습니다.
            response = pandas_df_agent.run(st.session_state.messages, callbacks=[stream_handler])
    
            # Assistant의 응답을 대화 기록에 추가하고 출력합니다.
            st.session_state.messages.append({"role": "assistant", "content": response})
            #st.write(response)
            # st.markdown(response)
    

