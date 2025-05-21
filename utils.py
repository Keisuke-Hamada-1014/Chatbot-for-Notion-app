"""
このファイルは、画面表示以外の様々な機能を提供する関数を定義するファイルです。
"""

############################################################
# 1. ライブラリの読み込み
############################################################
# streamlitアプリの表示を担当するモジュール
import streamlit as st
# ログ出力を行うためのモジュール
import logging
# 定数ファイルをインポート
import constants as ct
# LangChainのRAGモジュールをインポート
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
# JSONデータを扱うためのモジュール
import json

# ロガーの設定
logger = logging.getLogger(ct.LOGGER_NAME)


############################################################
# 2. ユーティリティ関数
############################################################

def build_error_message(message):
    """
    エラーメッセージを整形する関数
    
    Args:
        message: 表示するエラーメッセージの本文
        
    Returns:
        str: 整形されたエラーメッセージ
    """
    return f"エラーが発生しました: {message}\n管理者にお問い合わせください。"


def get_llm_response(query):
    """
    ユーザーの質問に対してLLMの回答を取得する関数
    
    Args:
        query: ユーザーからの質問文
        
    Returns:
        dict: LLMからの回答と参照情報を含む辞書
    """
    # セッション状態からLLMとRetrieverを取得
    llm = st.session_state.llm
    retriever = st.session_state.retriever
    
    # モードに応じたプロンプトテンプレートを選択
    if st.session_state.mode == ct.ANSWER_MODE_1:
        # 社内文書検索モード用プロンプト
        prompt_template = PromptTemplate.from_template(ct.SEARCH_PROMPT_TEMPLATE)
    else:
        # 社内問い合わせモード用プロンプト
        prompt_template = PromptTemplate.from_template(ct.CONTACT_PROMPT_TEMPLATE)
    
    # Retrieverを使って関連ドキュメントを取得
    retrieval_results = retriever.invoke(query)
    logger.info(f"検索結果: {len(retrieval_results)}件のドキュメントが見つかりました")
    
    # 検索結果からソース情報を抽出
    sources = []
    for doc in retrieval_results:
        # メタデータからソース情報を取得
        metadata = doc.metadata
        source_info = {
            "name": metadata.get("title", "不明なタイトル"),
            "url": metadata.get("url", ""),
            "page": metadata.get("page", ""),
            "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
        }
        sources.append(source_info)
    
    # コンテキストを構築
    context = "\n\n".join([doc.page_content for doc in retrieval_results])
    
    # 検索結果がない場合の処理
    if not context:
        logger.warning("検索結果が見つかりませんでした")
        return {
            "answer": "申し訳ありませんが、ご質問に関連する情報が見つかりませんでした。\n質問の表現を変えるか、別のトピックについてお尋ねください。",
            "sources": []
        }
    
    # プロンプトへの入力を準備
    prompt_input = {
        "context": context,
        "question": query
    }
    
    # プロンプトを実行してLLMからの回答を取得
    try:
        # プロンプトからLLMへの入力を生成
        prompt_content = prompt_template.format(**prompt_input)
        logger.debug(f"プロンプト: {prompt_content}")
        
        # LLMで回答を生成
        answer = llm.invoke(prompt_content)
        answer_text = answer.content
        
        # モードに応じた回答処理
        if st.session_state.mode == ct.ANSWER_MODE_1:
            # 社内文書検索モードの場合、関連文書の一覧を返す
            return {
                "answer": answer_text,
                "sources": sources
            }
        else:
            # 社内問い合わせモードの場合、質問への回答と参照元を返す
            return {
                "answer": answer_text,
                "sources": sources
            }
            
    except Exception as e:
        logger.error(f"LLM呼び出しエラー: {e}")
        raise Exception(f"LLMからの回答取得に失敗しました: {e}")