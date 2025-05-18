"""
このファイルは、アプリケーション全体で使用するユーティリティ関数を定義するモジュールです。
"""

############################################################
# 1. ライブラリの読み込み
############################################################
import logging
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import Chroma
import constants as ct


############################################################
# 2. ユーティリティ関数
############################################################
def build_error_message(base_message):
    """
    エラーメッセージを構築する関数
    
    Args:
        base_message: 基本のエラーメッセージ
        
    Returns:
        str: 整形されたエラーメッセージ
    """
    return f"{base_message}\n{ct.ERROR_CONTACT_MESSAGE}"


def get_llm_response(query):
    """
    ユーザーのクエリに対するLLMからの回答を取得する関数
    
    Args:
        query: ユーザーの質問文字列
        
    Returns:
        dict: LLMからの回答（辞書形式）
    """
    # 現在のアプリケーションモードを取得
    mode = st.session_state.mode
    
    # セッション状態からベクターストアを取得
    vectorstore = st.session_state.vectorstore
    
    # LLMの設定
    llm = ChatOpenAI(
        model_name=ct.LLM_MODEL_NAME,
        temperature=ct.LLM_TEMPERATURE
    )
    
    # 検索モードに応じた処理
    if mode == ct.ANSWER_MODE_1:
        # 社内文書検索モード：関連ドキュメントの検索
        docs = vectorstore.similarity_search(query, k=ct.TOP_K_DOCUMENTS)
        
        # 関連ドキュメントから回答を構築
        sources = []
        for doc in docs:
            metadata = doc.metadata
            sources.append({
                "name": metadata.get("title", "不明"),
                "url": metadata.get("url", "#"),
                "page": metadata.get("page", ""),
                "content": doc.page_content
            })
        
        return {
            "answer": "以下の社内文書が関連しています：",
            "sources": sources
        }
    
    elif mode == ct.ANSWER_MODE_2:
        # 社内問い合わせモード：LLMを使用した回答生成
        # 会話履歴の準備
        chat_history = []
        if "messages" in st.session_state:
            for i in range(0, len(st.session_state.messages), 2):
                if i+1 < len(st.session_state.messages):
                    chat_history.append((
                        st.session_state.messages[i]["content"],
                        st.session_state.messages[i+1]["content"]
                    ))
        
        # 検索と回答生成の連携
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": ct.TOP_K_DOCUMENTS}
        )
        
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )
        
        # 質問に対する回答を生成
        result = chain.invoke({"question": query, "chat_history": chat_history})
        
        # 回答と参照ソースを構築
        sources = []
        for doc in result["source_documents"]:
            metadata = doc.metadata
            sources.append({
                "name": metadata.get("title", "不明"),
                "url": metadata.get("url", "#"),
                "page": metadata.get("page", ""),
                "content": doc.page_content
            })
        
        return {
            "answer": result["answer"],
            "sources": sources
        }
    
    # デフォルトの応答
    return {
        "answer": "すみません、応答の生成中にエラーが発生しました。",
        "sources": []
    }