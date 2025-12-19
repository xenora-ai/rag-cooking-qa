# src/chunking/preprocess.py
import re


def clean_text(text: str) -> str:
    """
    Легка очистка тексту:
    - Видаляє зайві пробіли, нові рядки
    - Видаляє HTML-теги, якщо є
    - Видаляє дублікати пробілів
    """
    # Видаляємо HTML-теги
    text = re.sub(r'<[^>]+>', '', text)

    # Замінюємо кілька пробілів або табуляцій на один пробіл
    text = re.sub(r'\s+', ' ', text)
    return text.strip()  # Видаляємо пробіли на початку та кінці
