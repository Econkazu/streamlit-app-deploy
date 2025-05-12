# .envファイルから環境変数を読み込む
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage



# 専門家の種類とそれぞれのシステムメッセージを定義
EXPERTS = {
    "医療アドバイザー": {
        "name": "医療アドバイザー",
        "system_message": """あなたは経験豊富な医療専門家です。
医学的観点から正確で信頼性の高い情報を提供してください。
一般的な健康相談に答えますが、個別の医療診断や治療の推奨は避けてください。
必要に応じて医療機関への受診を勧めてください。"""
    },
    "金融アナリスト": {
        "name": "金融アナリスト",
        "system_message": """あなたは経験豊富な金融アナリストです。
投資、経済、金融商品、資産運用などについて専門的な分析と見解を提供してください。
リスクについても適切に説明し、一般的な投資助言に留めてください。
個別の投資推奨は避け、自己責任での判断を促してください。"""
    },
    "法律顧問": {
        "name": "法律顧問",
        "system_message": """あなたは法律の専門家です。
日本の法律に基づいた一般的な法的情報を提供してください。
個別の法的判断は避け、一般論として回答してください。
具体的な法的問題については、必ず専門の弁護士への相談を勧めてください。"""
    },
    "心理カウンセラー": {
        "name": "心理カウンセラー",
        "system_message": """あなたは経験豊富な心理カウンセラーです。
共感的で支持的な姿勢で、相談者の気持ちに寄り添いながら回答してください。
精神的な悩みについて一般的なアドバイスを提供しますが、
深刻な症状がある場合は専門医への相談を勧めてください。"""
    },
    "ビジネスコンサルタント": {
        "name": "ビジネスコンサルタント",
        "system_message": """あなたは経験豊富なビジネスコンサルタントです。
企業経営、戦略立案、マーケティング、組織運営などについて実践的なアドバイスを提供してください。
具体的な事例や分析手法を用いて、実行可能な提案を行ってください。"""
    },
    "技術エンジニア": {
        "name": "技術エンジニア",
        "system_message": """あなたは経験豊富なソフトウェアエンジニアです。
プログラミング、システム設計、最新技術などについて技術的な説明を提供してください。
コード例や実装方法についても具体的にアドバイスしてください。
ベストプラクティスやセキュリティ面での注意点も含めてください。"""
    }
}

def get_llm_response(user_input: str, selected_expert: str) -> str:
    """
    選択された専門家として、LLMからの回答を取得する関数
    
    Args:
        user_input (str): ユーザーからの質問
        selected_expert (str): 選択された専門家の種類
    
    Returns:
        str: LLMからの回答
    """
    try:
        # LangChainのChatOpenAIを初期化
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=1000
        )
        
        # 選択された専門家に対応するシステムメッセージを取得
        system_message = EXPERTS[selected_expert]["system_message"]
        
        # メッセージのリストを作成
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=user_input)
        ]
        
        # LLMの実行
        response = llm.invoke(messages)
        
        return response.content
    
    except Exception as e:
        return f"エラーが発生しました: {str(e)}"

def main():
    # ページの設定
    st.set_page_config(
        page_title="専門家選択型LLMアプリ",
        page_icon="🎯",
        layout="wide"
    )
    
    # タイトルとヘッダー
    st.title("🎯 専門家選択型LLMアドバイザー")
    st.markdown("**必要な専門家を選択して、専門的なアドバイスを受けられます**")
    
    # サイドバーに専門家選択のラジオボタンを配置
    with st.sidebar:
        st.header("専門家を選択")
        selected_expert = st.radio(
            "質問する専門家を選んでください：",
            list(EXPERTS.keys()),
            index=0
        )
        
        # 選択された専門家の説明を表示
        st.markdown("---")
        st.subheader(f"選択中: {EXPERTS[selected_expert]['name']}")
        st.write("この専門家として回答します。")
    
    # メインエリア
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("💬 質問を入力")
        user_input = st.text_area(
            "質問内容:",
            placeholder=f"{EXPERTS[selected_expert]['name']}に質問したいことを入力してください...",
            height=150
        )
        
        # 送信ボタン
        if st.button("📤 送信", type="primary"):
            if user_input.strip():
                with st.spinner("回答を生成中..."):
                    # LLMからの回答を取得
                    response = get_llm_response(user_input, selected_expert)
                    
                    # 回答をセッション状態に保存
                    if "responses" not in st.session_state:
                        st.session_state.responses = []
                    
                    st.session_state.responses.append({
                        "expert": EXPERTS[selected_expert]['name'],
                        "question": user_input,
                        "answer": response
                    })
            else:
                st.warning("質問を入力してください。")
    
    with col2:
        st.subheader("💡 回答")
        
        # 最新の回答を表示
        if "responses" in st.session_state and st.session_state.responses:
            latest_response = st.session_state.responses[-1]
            
            # 専門家の種類を表示
            st.info(f"**{latest_response['expert']}** からの回答:")
            
            # 質問と回答を表示
            st.markdown("**質問:**")
            st.markdown(f"_{latest_response['question']}_")
            
            st.markdown("**回答:**")
            st.markdown(latest_response['answer'])
        else:
            st.markdown("左側から質問を送信すると、ここに回答が表示されます。")
    
    # 履歴表示
    if "responses" in st.session_state and len(st.session_state.responses) > 1:
        st.markdown("---")
        st.subheader("📚 会話履歴")
        
        # 履歴をアコーディオンで表示
        for i, response in enumerate(reversed(st.session_state.responses[:-1])):
            with st.expander(f"{i+1}. {response['expert']}: {response['question'][:50]}..."):
                st.markdown(f"**専門家:** {response['expert']}")
                st.markdown(f"**質問:** {response['question']}")
                st.markdown(f"**回答:** {response['answer']}")
    
    # フッター
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #888; margin-top: 50px;'>
            専門的な回答を提供しますが、重要な決定には必ず適切な専門家に相談してください。
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()