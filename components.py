"""
このファイルは、アプリの画面表示に関する関数を定義するファイルです。
"""

############################################################
# 1. ライブラリの読み込み
############################################################
# streamlitアプリの表示を担当するモジュール
import streamlit as st
# ログ出力を行うためのモジュール
import logging
# マークダウンテキストを処理するためのモジュール
import re
# 定数ファイルをインポート
import constants as ct
# ユーティリティ関数をインポート
import utils

# ロガーの設定
logger = logging.getLogger(ct.LOGGER_NAME)


############################################################
# 2. 画面表示関数
############################################################

def display_app_title():
    """アプリのタイトルを表示する関数"""
    st.title(ct.APP_NAME)
    st.markdown(ct.APP_DESCRIPTION)


def display_select_mode():
    """モード選択を表示する関数"""
    # カラム分割
    col1, col2 = st.columns(2)
    
    # モード選択ラジオボタン
    with col1:
        selected_mode = st.radio(
            "モードを選択してください",
            options=[ct.ANSWER_MODE_1, ct.ANSWER_MODE_2],
            index=0 if st.session_state.mode == ct.ANSWER_MODE_1 else 1,
            horizontal=True
        )
    
    # モードの説明表示
    with col2:
        if selected_mode == ct.ANSWER_MODE_1:
            st.info(ct.ANSWER_MODE_1_DESCRIPTION)
        else:
            st.info(ct.ANSWER_MODE_2_DESCRIPTION)
    
    # モード変更を検知したらセッション変数を更新
    if selected_mode != st.session_state.mode:
        st.session_state.mode = selected_mode
        # モード変更ログの出力
        logger.info(f"モードが変更されました: {selected_mode}")
        # モード変更時は会話ログをクリア
        st.session_state.messages = []
        # 画面を更新
        st.rerun()


def display_initial_ai_message():
    """初期メッセージを表示する関数（会話ログが空の場合のみ）"""
    if len(st.session_state.messages) == 0:
        # モードに応じた初期メッセージを表示
        if st.session_state.mode == ct.ANSWER_MODE_1:
            initial_message = ct.INITIAL_MESSAGE_MODE_1
        else:
            initial_message = ct.INITIAL_MESSAGE_MODE_2
        
        # 初期メッセージをセッション変数に追加
        st.session_state.messages.append({"role": "assistant", "content": initial_message})


def display_conversation_log():
    """会話ログを表示する関数"""
    # セッション変数からメッセージを取得して表示
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def display_search_llm_response(llm_response):
    """
    社内文書検索モードにおけるLLM回答を表示する関数
    
    Args:
        llm_response: LLMからの回答（辞書形式）
        
    Returns:
        str: 表示用の整形されたコンテンツ
    """
    # 検索モードの場合の表示内容
    answer = llm_response.get("answer", "回答が見つかりませんでした。")
    sources = llm_response.get("sources", [])
    
    # 回答を表示
    st.markdown("### 関連文書")
    st.markdown(answer)
    
    # 参照元を表示
    if sources:
        st.markdown("### 参照元")
        for i, source in enumerate(sources, 1):
            source_name = source.get("name", "不明")
            source_url = source.get("url", "#")
            source_page = source.get("page", "")
            page_info = f" (ページ: {source_page})" if source_page else ""
            
            st.markdown(f"{i}. [{source_name}]({source_url}){page_info}")
    
    # 回答と参照元を結合して返す（ログ用）
    content = f"{answer}\n\n**参照元:**\n"
    for i, source in enumerate(sources, 1):
        source_name = source.get("name", "不明")
        source_url = source.get("url", "#")
        content += f"{i}. {source_name} ({source_url})\n"
    
    return content


def display_contact_llm_response(llm_response):
    """
    社内問い合わせモードにおけるLLM回答を表示する関数
    
    Args:
        llm_response: LLMからの回答（辞書形式）
        
    Returns:
        str: 表示用の整形されたコンテンツ
    """
    # 問い合わせモードの場合の表示内容
    answer = llm_response.get("answer", "回答が見つかりませんでした。")
    sources = llm_response.get("sources", [])
    
    # 回答を表示
    st.markdown("### 回答")
    st.markdown(answer)
    
    # 参照元を表示
    if sources:
        with st.expander("参照元", expanded=False):
            for i, source in enumerate(sources, 1):
                source_name = source.get("name", "不明")
                source_url = source.get("url", "#")
                source_page = source.get("page", "")
                source_content = source.get("content", "")
                
                st.markdown(f"**参照元 {i}: [{source_name}]({source_url})**")
                if source_page:
                    st.markdown(f"ページ: {source_page}")
                if source_content:
                    st.markdown("内容抜粋:")
                    st.markdown(source_content)
                st.divider()
    
    # 回答と参照元を結合して返す（ログ用）
    content = f"{answer}\n\n**参照元:**\n"
    for i, source in enumerate(sources, 1):
        source_name = source.get("name", "不明")
        source_url = source.get("url", "#")
        content += f"{i}. {source_name} ({source_url})\n"
    
    return content