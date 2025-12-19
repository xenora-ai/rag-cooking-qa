import gradio as gr
from src.pipeline.rag_pipeline import RAGPipeline


def rag_respond(query: str, groq_api_key: str, retriever_name: str, use_reranker: bool):
    pipeline = RAGPipeline(
        chunks_dir="data/chunks",
        doc_metadata_dir="data/data.csv",
        groq_api_key=groq_api_key,
        use_reranker=use_reranker
    )

    answer, used_docs = pipeline.run(
        query=query,
        retriever_name=retriever_name,
        top_n=15,
        top_k=5
    )

    # Формуємо Markdown-лінки
    sources_md = "\n".join(
        f"[{doc_id}]({pipeline.doc_metadata[doc_id]['url']})"
        for doc_id in used_docs
    )

    md_text = f"**Відповідь:**\n\n{answer}\n\n**Джерела:**\n{sources_md}"
    return md_text


with gr.Blocks() as demo:
    # Бокова панель для ключа та налаштувань
    with gr.Sidebar():
        gr.Markdown("## Введіть ваш GROQ API ключ")
        api_key_input = gr.Textbox(
            lines=1,
            placeholder="GROQ API ключ",
            type="password"
        )
        gr.Markdown("## Налаштування ретріверу")
        retriever_input = gr.Dropdown(
            choices=["bm25", "dense", "combined"],
            value="combined",
            label="Режим retriever"
        )
        reranker_input = gr.Checkbox(
            label="Увімкнути реранкер",
            value=True
        )

    # Основний блок UI
    with gr.Accordion("Про систему", open=False):
        gr.Markdown(
            """
            ## RAG Cooking QA
            
            Цей застосунок є демонстрацією Retrieval-Augmented Generation (RAG) системи
            для пошуку кулінарних ідей та рецептів.
            
            Система дозволяє ставити природномовні запити та отримувати відповіді,
            сформовані виключно на основі релевантних джерел, з обовʼязковими посиланнями
            на використані документи.
            
            ---
            
            ### Дані
            
            База знань сформована з **130 веб-сторінок**, присвячених кулінарії
            (сніданки, обіди, вечері, святкові страви, піца, борщ, салати, страви з картоплі,
            вареники тощо).
            
            Процес підготовки даних:
            - HTML-сторінки очищені від тегів і шуму
            - збережені у форматі `.txt`
            - поділені на текстові чанки
            - кожен документ має метадані (ID, URL, категорія)
            
            ---
            
            ### Пошук (Retrieval)
            
            - **BM25 (keyword search)** — ~5 секунд
            - **Semantic search (dense retriever)** — до 1 хвилини
            - **Combined retriever** — комбінація BM25 та dense (0.5 / 0.5)
            
            ---
            
            ### Reranker
            
            - retriever → топ-15 чанків  
            - reranker → топ-5  
            - можна увімкнути або вимкнути
            
            ---
            
            ### Генерація відповіді
            
            Використовується модель **llama-3.3-70b-versatile** через **Groq API**.
            Користувач вводить власний API-ключ.
            
            ---
            
            ### Як користуватися
            
            1. Створіть API-ключ Groq
            2. Введіть ключ
            3. Оберіть retriever
            4. Увімкніть/вимкніть реранкер
            5. Введіть запит
            """
        )

    query_input = gr.Textbox(
        lines=2,
        placeholder="Введіть свій запит..."
    )

    output = gr.Markdown(label="Відповідь")
    submit_btn = gr.Button("Отримати відповідь")

    # Виклик функції при натисканні
    submit_btn.click(
        rag_respond,
        inputs=[query_input, api_key_input, retriever_input, reranker_input],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch()
