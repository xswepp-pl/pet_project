import json
import sqlite3
import hashlib
import logging
import torch
import config
import threading
import re
import pandas as pd

import telebot
from telebot.util import smart_split

from typing import List, Optional, Sequence, Annotated
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from transformers import AutoTokenizer

from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchAny
from FlagEmbedding import FlagReranker

import warnings 
warnings.filterwarnings( "ignore", message=".*Local mode is not recommended for collections with more than 20,000 points.*" )
warnings.filterwarnings("ignore", message="You're using a XLMRobertaTokenizerFast tokenizer.*")


# =========================
# Логирование
# =========================
logging.basicConfig(
    level=logging.INFO,                                # Устанавливаем минимальный уровень логирования: INFO
    format="%(asctime)s [%(levelname)s] %(message)s",  # Формат вывода сообщений
    handlers=[logging.StreamHandler()]                 # Обработчик вывода логов
)


# =========================
# Загрузка базы аббревиатур
# =========================
# Подготовка базы: добавляем нормализованную колонку
ABBREVIATIONS_DF = pd.read_excel("documents/abbreviations.xlsx", header=0)

logging.info(f"База аббревиатур загружена: {len(ABBREVIATIONS_DF)} записей")

def find_abbreviation_expansions(abbreviations: list[str]) -> list[str]:
    """
    Ищет ВСЕ расшифровки аббревиатур в базе ABBREVIATIONS_DF.

    Parameters
    ----------
    abbreviations : list[str]
        Список аббревиатур для поиска.

    Returns
    -------
    list[str]
        Список строк вида "ABBR: расшифровка".
        Если вариантов несколько — все будут включены.
        Если нет ни одного — добавляется строка "ABBR: расшифровка не найдена".
    """
    if not abbreviations:
        return []

    ABBREVIATIONS_DF["abbr"] = ABBREVIATIONS_DF["abbr"].str.upper()

    results = []
    for abbr in abbreviations:
        abbr_upper = abbr.upper()
        matches = ABBREVIATIONS_DF[ABBREVIATIONS_DF["abbr"] == abbr_upper]

        if matches.empty:
            results.append(f"{abbr}: расшифровка не найдена")
            continue

        for _, row in matches.iterrows():
            results.append(f"{abbr}: {row['definition']}")

    return results


# =========================
# Кеш (SQLite)
# =========================
DB_FILE = "cache.db"

def init_db():
    """
    Инициализирует базу данных SQLite.
    Создаёт таблицу 'cache', если она ещё не существует.
    """
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            hash TEXT PRIMARY KEY,
            question TEXT,
            answer TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            user_id INTEGER,
            role TEXT CHECK(role IN ('human','assistant')),
            content TEXT,
            ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def get_hash(text: str) -> str:
    """Вычисляет SHA-256 хэш для заданного текста."""
    cleaned = re.sub(r"\s+", " ", text).strip().lower()
    return hashlib.sha256(cleaned.encode("utf-8")).hexdigest()

def get_answer(question: str) -> Optional[str]:
    """Получает сохранённый ответ из базы данных по заданному вопросу."""
    h = get_hash(question)
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("SELECT answer FROM cache WHERE hash=?", (h,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else None

def add_answer(question: str, answer: str):
    """Добавляет или обновляет ответ в базе данных для заданного вопроса."""
    h = get_hash(question)
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO cache (hash, question, answer) VALUES (?, ?, ?)",
        (h, question, answer)
    )
    conn.commit()
    conn.close()

def save_message(user_id: int, role: str, content: str):
    """Сохраняем только human/assistant сообщения"""
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("INSERT INTO messages (user_id, role, content) VALUES (?, ?, ?)", (user_id, role, content))
    conn.commit()
    conn.close()

def load_history(user_id: int, limit: int = 10) -> List[BaseMessage]:
    """Загружаем последние сообщения пользователя и агента"""
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("SELECT role, content FROM messages WHERE user_id=? ORDER BY ts DESC LIMIT ?", (user_id, limit))
    rows = cur.fetchall()
    conn.close()

    messages = []  # восстанавливаем порядок
    for role, content in reversed(rows):
        if role == "human":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages


# =========================
# Модель LLM и эмбеддинги
# =========================
# Настройки для LM Studio
LM_STUDIO_URL = "http://127.0.0.1:1234/v1"
MODEL_NAME_LM_STUDIO = "t-lite-it@q4_k_m"

logging.info(f"Подключаемся к LM Studio: {LM_STUDIO_URL}, модель {MODEL_NAME_LM_STUDIO}")

# Инициализируем модель через интерфейс OpenAI
llm = ChatOpenAI(
    base_url=LM_STUDIO_URL,
    api_key="lm-studio",              # Ключ не требуется, но поле не должно быть пустым
    model_name=MODEL_NAME_LM_STUDIO,
    temperature=0,                    # Для медицины лучше 0 (детерминированность)
    streaming=False                   # Внутри LangGraph лучше использовать False для стабильности узлов
)
tokenizer = AutoTokenizer.from_pretrained("./my_tokenizer")

logging.info("Загружаем эмбеддинги и векторную базу")

# Инициализация эмбеддингов, Qdrant и Reranker
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
client = QdrantClient(path="qdrant_db")
reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True, device='cpu')


# =========================
# Состояние графа (Pydantic)
# =========================
class AgentState(BaseModel):
    is_medical_intent: bool = Field(default=True,
        description="Флаг, указывающий относится ли вопрос пользователя к медицине (True) или нет (False)"
    )
    user_id: Optional[int] = Field(default=None,
        description="Уникальный идентификатор пользователя Telegram для управления историей сообщений"
    ) 
    messages: Annotated[Sequence[BaseMessage], add_messages] = Field(default_factory=list,
        description="История сообщений диалога (HumanMessage, AIMessage), используется для контекстуального понимания запроса"
    )
    rewritten_question: Optional[str] = Field(default=None,
        description="Оптимизированная версия исходного вопроса пользователя, обогащенная медицинской терминологией и контекстом"
    )
    cache_hit: bool = Field(default=False,
        description="Флаг, указывающий был ли найден ответ в кэше (True) или требуется генерация (False)"
    )
    expanded_queries: List[str] = Field(default_factory=list,
        description="Список уточненных поисковых запросов, сгенерированных для более полного покрытия темы"
    )
    abbreviations: str = Field(default="",
        description="Медицинские аббревиатуры, обнаруженные в запросе пользователя и их расшифровки"
    )
    search_results_text: Optional[str] = Field(default=None,
        description="Агрегированный текст релевантных фрагментов клинических рекомендаций для генерации ответа"
    )
    answer: Optional[str] = Field(default=None,
        description="Сгенерированный ответ ассистента на основе клинических рекомендаций и контекста диалога"
    )


# =========================
# Узлы графа
# =========================
def classify_intent_node(state: AgentState) -> AgentState:
    """Классификатор темы запросов пользователля (медицинский/немедицинский)."""
    system_prompt = (
        "Ты — строгий классификатор запросов. Определи, относится ли последний вопрос пользователя к медицинской тематике.\n\n"
        "МЕДИЦИНСКИЕ ЗАПРОСЫ включают:\n"
            "- симптомы, жалобы, синдромы\n"
            "- диагнозы, заболевания, нозологии\n"
            "- анализы, обследования, диагностику\n"
            "- лекарства, дозировки, схемы лечения\n"
            "- операции, процедуры, реабилитацию\n"
            "- анатомию, физиологию, патофизиологию\n"
            "- медицинские приборы, исследования\n"
            "- медицинские аббревиатуры (АГ, ХСН, СД2 и т.д.)\n\n"

        "ОБЩИЕ ЗАПРОСЫ включают:\n"
            "- приветствия, small talk\n"
            "- вопросы о личности ассистента\n"
            "- бытовые вопросы, юмор, философию\n"
            "- технические вопросы, программирование\n"
            "- любые темы, не связанные с медициной\n\n"

        "Верни СТРОГО JSON без текста до или после:\n"
        "{\"is_medical_intent\": true} или {\"is_medical_intent\": false}"
    )
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=(
            f"История диалога:\n{"\n".join([m.content for m in state.messages[:-1]])}\n\n"
            f"Последний вопрос пользователя, требующий классификации:\n{state.messages[-1].content}"
        ))
    ]
    response = llm.invoke(messages).content
    
    try:
        match = re.search(r"\{.*\}", response, re.DOTALL).group(0)       # ищем первый JSON-массив в ответе
        intent = json.loads(match)                                       # превращаем JSON-строку в Python-список
        state.is_medical_intent = intent.get("is_medical_intent", True)  # получаем менку намерения пользователя от LLM

        logging.info(f"Узел (classify_intent_node). Обнаружен {"медицинский" if state.is_medical_intent else "немедицинский"} запрос.")

    except Exception as e:
        logging.error(f"Узел (classify_intent_node). Ошибка: {e}")
        state.is_medical_intent = True # по умолчанию считаем медицинским
    
    torch.cuda.empty_cache()
    return state

def detect_abbreviations_node(state: AgentState) -> AgentState:
    """Обнаружение медицинских аббревиатур в запросе пользователя."""
    system_prompt = (
        "Ты - медицинский эксперт. Проанализируй вопрос и выдели ВСЕ медицинские аббревиатуры и сокращения.\n\n"
        "ИНСТРУКЦИИ:\n"
        "- Выдели только медицинские аббревиатуры (например: ОРВИ, ХБП, СД, АГ, ИБС и другие)\n"
        "- Игнорируй общеупотребительные сокращения (например: и т.д., и др.)\n"
        "- Включай как русские, так и латинские медицинские аббревиатуры\n"
        "- Если аббревиатур нет - верни пустой JSON массив\n"
        "- Верни результат СТРОГО в формате JSON массива строк, например: [\"ОРВИ\", \"АГ\", \"ИБС\"]\n\n"

        "ПРИМЕРЫ:\n"
        "Вопрос: 'диагностика АГ и лечение СД 2 типа' → [\"АГ\", \"СД\"]\n"
        "Вопрос: 'как лечить ОРВИ у детей?' → [\"ОРВИ\"]\n"
        "Вопрос: 'что такое ХБП 3 стадии?' → [\"ХБП\"]\n"
        "Вопрос: 'обычная простуда' → []"
    )
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=state.messages[-1].content)
    ]
    response = llm.invoke(messages).content
    
    try:
        match = re.search(r"\[.*?\]", response, re.DOTALL).group(0)                  # ищем первый JSON-массив в ответе
        abbreviations = json.loads(match)                                            # Превращаем JSON-строку в Python-список
        abbreviations_with_expansions = find_abbreviation_expansions(abbreviations)  # Выполняем поиск расшифровок аббревиатур и формируем словарь
        state.abbreviations = "РАСШИФРОВКИ АББРЕВИАТУР:\n" + "\n".join(abbreviations_with_expansions)

        logging.info(f"Узел (detect_abbreviations_node). Обнаружены аббревиатуры: {abbreviations_with_expansions}")

    except Exception as e:
        logging.error(f"Узел (detect_abbreviations_node). Ошибка: {e}")

    torch.cuda.empty_cache()
    return state

def rewrite_node(state: AgentState) -> AgentState:
    """Переформулировка последнего вопроса пользователя с учётом контекста истории и аббревиатур."""
    system_prompt = (
        "Ты - медицинский эксперт. Проанализируй вопрос и переформулируй его для эффективного поиска в клинических рекомендациях.\n\n"
        "ПЕРЕФОРМУЛИРУЙ ЕСЛИ:\n"
            "- Вопрос короче 3 слов или слишком общий\n"
            "- Есть разговорные/неформальные формулировки\n"
            "- Не хватает медицинского контекста\n"
            "- Есть орфографические ошибки в терминах\n\n"

        "НЕ МЕНЯЙ ЕСЛИ:\n"
            "- Вопрос уже содержит специфические медицинские термины\n"
            "- Четко сформулирован с достаточным контекстом\n"
            "- Не касается медицинской тематики\n\n"

        "ПРАВИЛА:\n"
            "- Сохраняй исходный смысл, не добавляй дополнительный смысл.\n"
            "- Используй стандартные медицинские термины.\n"
            "- Добавь недостающий контекст (симптомы, возраст, длительность если уместно).\n"
            "- Верни ТОЛЬКО переформулированный вопрос без пояснений.\n"
            "- Если вопрос хороший - верни его без изменений."
    )

    # Добавляем информацию об аббревиатурах, если она есть
    if state.abbreviations:
        system_prompt += f"\n\nВОЗМОЖНЫЕ РАСШИФРОВКИ АББРЕВИАТУР, НАЙДЕННЫХ В ВОПРОСЕ ПОЛЬЗОВАТЕЛЯ:\n{state.abbreviations}."

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=(
            f"История диалога:\n{"\n".join([m.content for m in state.messages[:-1]])}\n\n"
            f"Переформулируй последний вопрос пользователя:\n{state.messages[-1].content}"))
    ]
    # input_tokens = len(tokenizer.encode("\n".join([m.content for m in state.messages[:-1]]) + state.messages[-1].content + system_prompt))
    # logging.info(f"Узел (rewrite_node). На вход подано {input_tokens} ({input_tokens*100/16384:.2f}%) токенов.")

    response = llm.invoke(messages).content
    state.rewritten_question = response

    logging.info(f"Узел (rewrite_node). Переформулировка: '{state.messages[-1].content}' -> '{response}'.")
    
    torch.cuda.empty_cache()
    return state

def check_cache_node(state: AgentState) -> AgentState:
    """Поиск вопроса в кэше."""
    cached = get_answer(state.rewritten_question)
    if cached:
        state.answer = cached
        state.cache_hit = True
    else:
        state.cache_hit = False
    return state

def expand_node(state: AgentState) -> AgentState:
    """Генерация уточняющих вопросов для улучшения поиска в клинических рекомендациях."""
    max_questions = 3
    system_prompt = (
        "Ты — медицинский информационный специалист. Сгенерируй поисковые запросы для поиска по базе клинических рекомендаций.\n\n"

        "ЦЕЛЬ: Найти максимально релевантные документы в медицинской базе\n\n"

        "ПРАВИЛА СОЗДАНИЯ ЗАПРОСОВ:\n"
            f"- Подготовь до {max_questions} уточняющих запросов по вопросу пользователя.\n"
            "- Используй ТОЛЬКО официальные медицинские термины\n"
            "- Включай синонимы и аббревиатуры (АГ → артериальная гипертензия)\n"
            "- Фокусируйся на ключевых словах, а не на полных вопросах\n"
            "- Разделяй сложные темы на конкретные аспекты\n"
            "- Указывай конкретные нозологии, препараты, процедуры\n"
            "- Используй термины, которые могут быть в заголовках разделов рекомендаций\n\n"

        "ЧЕГО ИЗБЕГАТЬ:\n"
            "- Вопросительных форм ('как лечить?' → 'лечение')\n"
            "- Общих фраз ('что делать при' → конкретное состояние)\n"
            "- Разговорных выражений\n"
            "- Лишних слов\n\n"

        "Верни результат строго в формате JSON массива строк и НЕ добавляй никаких пояснений, текста до или после, например:\n"
        "[\"вопрос1\", \"вопрос2\", \"вопрос3\"]"
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=state.rewritten_question)
    ]
    response = llm.invoke(messages).content

    try:
        match = re.search(r"\[.*?\]", response, re.DOTALL).group(0)  # ищем первый JSON-массив в ответе
        queries = json.loads(match)[:max_questions]                  # Превращаем JSON-строку в Python-список
        state.expanded_queries = queries or [state.rewritten_question]

        logging.info(f"Узел (expand_node). Сгенерировано {len(queries)} запросов.")

    except Exception as e:
        queries = []
        logging.error(f"Узел (expand_node). Ошибка: {e}")
    
    torch.cuda.empty_cache()
    return state

def retrieve_node(state: AgentState) -> AgentState:
    """Иерархический поиск в Qdrant с реранжированием BGE."""
    k_docs = 3        # Сколько документов (КР) искать на первом этапе
    k_chunks = 100    # Сколько фрагментов искать внутри этих документов
    max_tokens = 1200
    
    final_blocks = []  # Список для накопления результатов поиска
    for query in state.expanded_queries:

        logging.info(f"Узел (retrieve_node). Обрабатываем вопрос --> {query}")

        query_vector = embeddings.embed_query(f"query: {query}")

        # ЭТАП 1: Поиск релевантных документов (Parent Search)
        search_docs = client.query_points(
            collection_name="clin_rec_docs", 
            query=query_vector, 
            limit=k_docs
        ).points

        found_doc_ids = [str(d.payload['doc_id']) for d in search_docs]  # Извлекаем doc_id найденных документов

        logging.info(f"Узел (retrieve_node). Вырбаны документы: {[d.payload['name'][:30] for d in search_docs]}")

        # ЭТАП 2: Поиск чанков внутри найденных документов (Child Search)
        search_chunks = client.query_points(
            collection_name="clin_rec_chunks",
            query=query_vector,
            limit=k_chunks,
            query_filter=Filter(must=[FieldCondition(key="metadata.doc_id", match=MatchAny(any=found_doc_ids))])
        ).points

        # ЭТАП 3: РЕРАНЖИРОВАНИЕ (Cross-Encoding)
        pairs = []  # Формируем пары [вопрос, текст_чанка]
        for hit in search_chunks:
            content = hit.payload.get('page_content', hit.payload.get('metadata', {}).get('page_content', ""))
            pairs.append([query, content])
        scores = reranker.compute_score(pairs)  # Получаем оценки релевантности от BGE модели
    
        # Присваиваем новые оценки чанкам
        for hit, score in zip(search_chunks, scores):
            hit.score = score # Заменяем векторный косинус на оценку реранкера

        # Сортируем по убыванию оценки реранкера
        search_chunks.sort(key=lambda x: x.score, reverse=True)

        # ЭТАП 4: Сборка контекста с учетом лимита токенов
        current_tokens = 0
        for chunk in search_chunks:
            text = f"Источник [{chunk.payload.get('metadata', {}).get('name', "")}]: {chunk.payload.get("page_content", "")[9:]}"
            chunk_tokens = len(tokenizer.encode(text))
  
            final_blocks.append(text)
            current_tokens += chunk_tokens

            if current_tokens + chunk_tokens > max_tokens:
                break
    
    state.search_results_text = "\n\n".join(final_blocks)

    logging.info(f"Узел (retrieve_node). Всего собрано контекста: {len(tokenizer.encode(state.search_results_text))} токенов.")
    
    return state

def medical_generate_node(state: AgentState) -> AgentState:
    """Генерация финального ответа на основе поисковых результатов."""
    system_prompt = (
        "Ты - виртуальный медицинский ассистент для помощи врачам. Ответь на вопрос строго на основе предоставленной информации.\n\n"

        "СТРОГИЕ ПРАВИЛА:\n"
            "ДЕЛАЙ: Используй только информацию из предоставленных клинических рекомендаций\n"
            "ДЕЛАЙ: Указывай конкретные цифры, дозировки, критерии из рекомендаций\n"
            "ДЕЛАЙ: Указывай четкие алгоритмы действий по шагам\n"
            "ДЕЛАЙ: Указывай конкретные диагностические критерии\n"
            "ДЕЛАЙ: Указывай точные названия препаратов и схемы лечения\n"
            "НЕЛЬЗЯ: Добавлять информацию не из предоставленных источников\n"
            "НЕЛЬЗЯ: Давать личные мнения или непроверенные данные\n"
            "НЕЛЬЗЯ: НЕ предлагай обратиться к врачу (это подразумевается)\n"
            "НЕЛЬЗЯ: Ставить диагнозы без указания на рекомендации\n\n"

        f"ИНФОРМАЦИЯ ДЛЯ ОТВЕТА ИЗ БАЗЫ ДАННЫХ КЛИНИЧЕСКИХ РЕКОМЕНДАЦИЙ:\n{state.search_results_text}\n\n"
        "Если информации недостаточно, честно скажи об этом."
    )
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=state.rewritten_question)
    ]
    
    input_tokens = len(tokenizer.encode(state.rewritten_question + system_prompt))
    logging.info(f"Узел (medical_generate_node). Генерация финального ответа. На вход подано {input_tokens} ({input_tokens*100/16384:.2f}%) токенов.")

    state.answer = llm.invoke(messages).content
    
    output_tokens = len(tokenizer.encode(state.answer))
    logging.info(f"Узел (medical_generate_node). Финальный ответ сгенерирован и составил {output_tokens} ({output_tokens*100/16384:.2f}%) токенов.")

    torch.cuda.empty_cache()
    return state

def no_medical_generate_node(state: AgentState) -> AgentState:
    system_prompt = (
        "Ты - виртуальный медицинский ассистент для помощи врачам. "
        "На немедицинские вопросы отвечай вежливо, но кратко. Если тебя просят сделать что-то не по теме "
        "(например, написать код), вежливо откажись, сказав, что ты специализируешься на медицине."
    )
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=(
            f"История диалога:\n{"\n".join([m.content for m in state.messages[:-1]])}\n\n"
            f"Последний вопрос пользователя:\n{state.messages[-1].content}"
        ))
    ]

    input_tokens = len(tokenizer.encode(state.messages[-1].content + system_prompt))
    logging.info(f"Узел (no_medical_generate_node). Генерация финального ответа. На вход подано {input_tokens} ({input_tokens*100/16384:.2f}%) токенов.")

    state.answer = llm.invoke(messages).content

    output_tokens = len(tokenizer.encode(state.answer))
    logging.info(f"Узел (no_medical_generate_node). Финальный ответ сгенерирован и составил {output_tokens} ({output_tokens*100/16384:.2f}%) токенов.")

    torch.cuda.empty_cache()
    return state

def save_cache_node(state: AgentState) -> AgentState:
    """Сохраняет ответ агента в кэше, если он новый (не найден ранее)."""
    add_answer(state.rewritten_question, state.answer)
    logging.info(f"Узел (save_cache_node). Ответ сохранён в кэше для вопроса: {state.rewritten_question}")
    return state


# =========================
# Построение графа
# =========================
graph = StateGraph(AgentState)

graph.set_entry_point("classify_intent")  # Начинаем с узла классификации темы сообщения

graph.add_node("classify_intent", classify_intent_node)
graph.add_node("detect_abbreviations", detect_abbreviations_node)
graph.add_node("rewrite", rewrite_node)
graph.add_node("check_cache", check_cache_node)
graph.add_node("expand", expand_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("medical_generate", medical_generate_node)
graph.add_node("save_cache", save_cache_node)
graph.add_node("no_medical_generate", no_medical_generate_node)

# Компоненты логики классификации темы сообщения и структура немедицинской ветки
def route_intent(state: AgentState):
    if state.is_medical_intent:  # Ветвление: если вопрос по медицине — медицинская ветка, иначе — немедицинская ветка
        return "detect_abbreviations"
    return "no_medical_generate"

graph.add_conditional_edges("classify_intent", route_intent,
    {
        "detect_abbreviations": "detect_abbreviations",
        "no_medical_generate": "no_medical_generate"
    }
)
graph.add_edge("no_medical_generate", END)  # Заканчиваем немедицинскую ветку

# Компоненты логики основной медицинской ветки
graph.add_edge("detect_abbreviations", "rewrite")
graph.add_edge("rewrite", "check_cache")

def route_after_cache(state: AgentState):
    return END if state.cache_hit else "expand"  # Ветвление: если кеш найден — конец, иначе идём дальше

graph.add_conditional_edges("check_cache", route_after_cache, 
    {
        "expand": "expand", 
        END: END  # Заканчиваем медицинскую ветку, и загружаем ответ из кэша
    }
)

graph.add_edge("expand", "retrieve")
graph.add_edge("retrieve", "medical_generate")
graph.add_edge("medical_generate", "save_cache")
graph.add_edge("save_cache", END)

memory = MemorySaver()
app = graph.compile(checkpointer=memory)


# =========================
# Telegram bot с интегрированным агентом
# =========================
bot = telebot.TeleBot(config.TOKEN)

# Система управления очередью запросов
lock = threading.Lock()
busy = False
current_user = None

@bot.message_handler(commands=['start'])
def welcome(message):
    """Обработчик команды /start"""
    bot.send_message(message.chat.id, "👋 Привет! Я — виртуальный медицинский ассистент Бинтик.")

@bot.message_handler(commands=['status'])
def status(message):
    """Проверка статуса бота"""
    with lock:
        status_text = f"🤖 Статус бота: {'Занят' if busy else 'Свободен'}"
        if busy:
            status_text += f"\nТекущий пользователь: {current_user}"
        bot.send_message(message.chat.id, status_text)

@bot.message_handler(content_types=['text'])
def handle_query(message):
    """Основной обработчик запросов с интегрированным агентом"""
    global busy, current_user
    user_id = message.chat.id
    user_question = message.text
    logging.info(f"Запрос от {user_id}: {user_question}")

    # Проверка занятости бота
    with lock:
        if busy and current_user != user_id:
            bot.send_message(user_id, "⚠️ Занят другим запросом. Попробуйте позже.")
            return
        busy = True
        current_user = user_id

    try:
        bot.send_message(user_id, "💭 Задачу принял, формулирую ответ...")

        final_state: AgentState = app.invoke(
            {"user_id": user_id, "messages": load_history(user_id, limit=3) + [HumanMessage(content=user_question)]},
            config={"configurable": {"thread_id": str(user_id)}}
        )

        answer = final_state.get("answer")  # Ответ Агента

        # Разбиваем на части и отправляем каждую
        for chunk in smart_split(answer, chars_per_string=4000):
            bot.send_message(user_id, chunk)

        save_message(user_id, "human", user_question)  # Сохраняем ответ пользователя в базу SQL
        save_message(user_id, "assistant", answer)     # Сохраняем ответ ассистента в базу SQL

    except Exception as e:
        bot.send_message(user_id, f"Произошла ошибка. Обратитесь к разработчикам.\n{e}")
        logging.error(f"Ошибка {e}")

    finally:
        with lock:
            busy = False
            current_user = None

if __name__ == "__main__":
    logging.info("Инициализация базы кеша...")
    init_db()
    logging.info("Запуск Telegram-бота...")
    bot.polling(non_stop=True)
