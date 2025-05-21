"""
このファイルは、アプリ起動時に実行される初期化処理を行う関数を定義するファイルです。
"""

############################################################
# 1. ライブラリの読み込み
############################################################
# 「.env」ファイルから環境変数を読み込むための関数
from dotenv import load_dotenv
# 環境変数を操作するモジュール
import os
# 日付・時刻を扱うためのモジュール
import datetime
# ログ出力を行うためのモジュール
import logging
# ログのフォーマットを設定するためのモジュール
from logging.handlers import RotatingFileHandler
# streamlitアプリの表示を担当するモジュール
import streamlit as st
# OpenAIのAPIを呼び出すためのモジュール
import openai
# LangChainのChatOpenAIを使用するためのモジュール
from langchain_openai import ChatOpenAI
# LangChainのPromptTemplateを使用するためのモジュール
from langchain.prompts import PromptTemplate
# LangChainのNotionDBLoaderを使用するためのモジュール
from langchain_community.document_loaders import NotionDBLoader
# LangChainのTextSplitterを使用するためのモジュール
from langchain_text_splitters import RecursiveCharacterTextSplitter
# LangChainのOpenAIEmbeddingsを使用するためのモジュール
from langchain_openai import OpenAIEmbeddings
# LangChainのChromaを使用するためのモジュール
from langchain_community.vectorstores import Chroma
# 固定値・変数を定義しているファイル
import constants as ct


############################################################
# 2. 初期化関数
############################################################
def initialize():
    """
    アプリ起動時に実行される初期化処理を行う関数
    - 環境変数の読み込み
    - 各種ディレクトリの作成
    - ログ設定
    - セッション変数の初期化
    - LLMの初期化
    - Notion連携の初期化
    - ベクターストアの初期化
    """
    # ==========================================
    # 2-1. 環境変数の読み込み
    # ==========================================
    # 「.env」ファイルから環境変数を読み込む
    load_dotenv()
    # OpenAI APIキーを環境変数から取得
    openai_api_key = os.getenv("OPENAI_API_KEY")
    # Notion統合トークンを環境変数から取得
    notion_integration_token = os.getenv("NOTION_INTEGRATION_TOKEN")
    # NotionデータベースIDを環境変数から取得
    notion_database_id = os.getenv("NOTION_DATABASE_ID")

    # APIキーの存在チェック
    if not openai_api_key:
        raise ValueError("OpenAI APIキーが設定されていません。.envファイルを確認してください。")
    if not notion_integration_token or not notion_database_id:
        raise ValueError("Notion APIの設定が不足しています。.envファイルを確認してください。")

    # ==========================================
    # 2-2. ディレクトリ作成
    # ==========================================
    # 各種ディレクトリの存在確認&なければ作成
    os.makedirs(ct.LOG_DIR, exist_ok=True)
    os.makedirs(ct.DATA_DIR, exist_ok=True)
    os.makedirs(ct.CHROMA_DIR, exist_ok=True)

    # ==========================================
    # 2-3. ログ設定
    # ==========================================
    # ロガーの取得
    logger = logging.getLogger(ct.LOGGER_NAME)
    # ログレベルの設定
    logger.setLevel(logging.INFO)
    # ログハンドラーの設定（ファイル出力）
    log_handler = RotatingFileHandler(
        filename=os.path.join(ct.LOG_DIR, "application.log"),
        maxBytes=ct.LOG_MAX_BYTES,
        backupCount=ct.LOG_BACKUP_COUNT,
        encoding="utf-8"
    )
    # ログのフォーマット設定
    log_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    # ハンドラーにフォーマットを設定
    log_handler.setFormatter(log_format)
    # ロガーにハンドラーを追加
    logger.addHandler(log_handler)

    # ==========================================
    # 2-4. セッション変数の初期化
    # ==========================================
    # セッション変数の初期化（初回のみ）
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # モード選択のセッション変数初期化（初回のみ）
    if "mode" not in st.session_state:
        st.session_state.mode = ct.ANSWER_MODE_1

    # ==========================================
    # 2-5. LLMの初期化
    # ==========================================
    # OpenAI ChatGPT APIクライアントの初期化
    st.session_state.llm = ChatOpenAI(
        api_key=openai_api_key,
        model=ct.MODEL_NAME,
        temperature=ct.TEMPERATURE,
        max_tokens=ct.MAX_TOKENS
    )

    # ==========================================
    # 2-6. Notion連携の初期化
    # ==========================================
    # NotionDBLoader インスタンスの初期化
    notion_loader = NotionDBLoader(
        integration_token=notion_integration_token,
        database_id=notion_database_id
    )

    # Notionからドキュメントを読み込み
    try:
        notion_docs = notion_loader.load()
        logger.info(f"Notionから{len(notion_docs)}件のドキュメントを読み込みました")
    except Exception as e:
        logger.error(f"Notionドキュメントの読み込みに失敗しました: {e}")
        raise ValueError(f"Notionドキュメントの読み込みに失敗しました: {e}")
    
    # テキスト分割の設定
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=ct.CHUNK_SIZE,
        chunk_overlap=ct.CHUNK_OVERLAP
    )
    
    # ドキュメントをチャンクに分割
    chunks = text_splitter.split_documents(notion_docs)
    logger.info(f"ドキュメントを{len(chunks)}個のチャンクに分割しました")

    # ==========================================
    # 2-7. ベクターストアの初期化
    # ==========================================
    # Embeddingモデルのインスタンス化
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    
    # Chromaベクターストアの初期化（または既存のものを読み込み）
    try:
        # 既存のベクターストアがあれば読み込み
        vectorstore = Chroma(
            persist_directory=ct.CHROMA_DIR,
            embedding_function=embeddings
        )
        
        # ベクターストア内のドキュメント数を確認
        collection_count = vectorstore._collection.count()
        logger.info(f"既存のベクターストアから{collection_count}件のドキュメントを読み込みました")
        
        # 更新が必要な場合は再構築
        if collection_count == 0 or os.getenv("REBUILD_VECTORSTORE", "false").lower() == "true":
            # ベクターストアを再構築
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=ct.CHROMA_DIR
            )
            vectorstore.persist()
            logger.info("ベクターストアを再構築しました")
    except Exception as e:
        # 初回またはエラー時は新規作成
        logger.info(f"ベクターストアを新規作成します: {e}")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=ct.CHROMA_DIR
        )
        vectorstore.persist()
        logger.info("ベクターストアを新規作成しました")
    
    # ベクターストアからRetrieverを作成
    st.session_state.retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": ct.RETRIEVER_K}
    )
    
    logger.info("初期化処理が完了しました")