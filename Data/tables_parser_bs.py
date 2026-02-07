import requests
import logging
import re
import json
from typing import Optional, List, Dict
from bs4 import BeautifulSoup, Comment

# Настройка базового логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =========================================================
# 1. ЗАГРУЗКА HTML-КОДА
# =========================================================

def fetch_html_content(url: str) -> Optional[str]:
    """Скачивает HTML-контент страницы."""
    try:
        logging.info(f"Попытка загрузки URL: {url}")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, timeout=15, headers=headers)
        response.raise_for_status()
        response.encoding = response.apparent_encoding
        logging.info("SUCCESS: Страница успешно загружена.")
        return response.text
    except requests.exceptions.RequestException as e:
        logging.error(f"Ошибка запроса к {url}: {e}")
        return None

# =========================================================
# 2. ОЧИСТКА ТЕКСТА
# =========================================================

def clean_html_content(html_content: str) -> str:
    """
    Очищает HTML, удаляет лишние теги и возвращает чистый текст.
    """
    soup = BeautifulSoup(html_content, 'lxml')
    
    # 1. Удаление служебных, нетекстовых тегов и комментариев
    REMOVAL_TAGS = ['script', 'style', 'noscript', 'iframe', 'meta', 'link', 'br']
    for element in soup(REMOVAL_TAGS):
        element.decompose()
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()
    
    # 2. Удаление навигации и футеров
    elements_to_remove = [
        soup.find('div', class_='block-0-menu-16'),
        soup.find('nav', id='menu'),
        soup.find('div', id='n'),
        soup.find('header', class_='landing-header'),
        soup.find('div', class_='landing-footer'),
        soup.find('style', type='text/css'),
    ]

    for element in elements_to_remove:
        if element:
            element.decompose()
            
    # 3. Извлечение текста
    main_content_tag = soup.find('div', class_='landing-main')
    if not main_content_tag:
        main_content_tag = soup.body if soup.body else soup

    for tag_name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div', 'section', 'li', 'hr', 'table']:
        for tag in main_content_tag.find_all(tag_name):
            tag.append('\n')

    pure_text = main_content_tag.get_text(separator=' ', strip=True)

    # 4. Финальная нормализация пробелов
    pure_text = re.sub(r'[\s]{2,}', '\n', pure_text)
    
    return pure_text.strip()


# =========================================================
# 3. ПАРСИНГ ДАННЫХ
# =========================================================

def get_program_level(code: str) -> str:
    """Определяет уровень образования по коду."""
    if re.match(r'\d{2}\.03\.\d{2}', code):
        return "Бакалавриат"
    if re.match(r'\d{2}\.05\.\d{2}', code):
        return "Специалитет"
    return "Другое"

def extract_structured_data(full_text: str) -> List[Dict]:
    """
    Парсит текст и извлекает данные по программам.
    """
    # 1. Отсекаем нижнюю таблицу (заголовки), которая ломает парсинг
    pre_filter_match = re.split(r'Наименование направления подготовки', full_text, 1, re.DOTALL)
    full_text_filtered = pre_filter_match[0].strip()
    
    # 2. Разбиваем текст на блоки программ
    program_blocks = re.findall(
        r'(\d{2}\.\d{2}\.\d{2}.*?)(?=\d{2}\.\d{2}\.\d{2}|$)', 
        full_text_filtered, re.DOTALL
    )
    
    structured_programs = []
    
    for block_content in program_blocks:
        if not block_content: 
            continue

        # Пропускаем блоки без ключевых слов (защита от мусора)
        if 'Форма обучения' not in block_content:
            continue

        # --- Код и Название ---
        # Код включает опциональный под-код (XX.XX.XX.XX)
        code_match = re.search(r'^(\d{2}\.\d{2}\.\d{2}(?:\.\d{2})?)', block_content)
        program_code = code_match.group(1) if code_match else "N/A"
        
        # Название: берем текст после кода до фразы "Форма обучения"
        name_match = re.search(r'^\d{2}\.\d{2}\.\d{2}(?:\.\d{2})?\s*(.*?)(?=Форма обучения)', block_content, re.DOTALL)
        program_name = name_match.group(1).strip() if name_match else "N/A"
        
        # --- Форма обучения ---
        form_match = re.search(r'Форма обучения:\s*([^\s]+)', block_content)
        study_form = form_match.group(1).strip() if form_match else "N/A"

        # --- Предметы ---
        subjects_match = re.search(
            r'Предметы:\s*(.*?)(?=\s*Количество мест:|Стоимость обучения)', 
            block_content, re.DOTALL
        )
        subjects_str = subjects_match.group(1).strip() if subjects_match else "N/A"
        
        # Очистка предметов (удаляем скобки с баллами, нормализуем разделители)
        subjects_clean_str = re.sub(r'\s*\([^)]*\)', '', subjects_str)
        subjects_final = re.sub(r'\s*([+/])\s*', r' \1 ', subjects_clean_str).replace('  ', ' ').strip()


        # --- Стоимость (РФ) ---
        # Ищем паттерн с защитой от опечаток в слове "для" (лдя/пропуск слова)
        cost_rf_match_explicit = re.search(r'Стоимость обучения\s*(?:[^\s]+\s*)?граждан РФ:\s*(\d+\s*\d+)', block_content)
        cost_rf = cost_rf_match_explicit.group(1).replace(' ', '') if cost_rf_match_explicit else "N/A"
        
        # Резервный поиск, если не нашли явного упоминания РФ
        if cost_rf == "N/A":
            cost_rf_match_general = re.search(r'Стоимость обучения:\s*(\d+\s*\d+)', block_content)
            if cost_rf_match_general:
                cost_rf = cost_rf_match_general.group(1).replace(' ', '')

        # --- Стоимость (Иностранцы) ---
        cost_foreign_match = re.search(r'Стоимость обучения для иностранных граждан:\s*(\d+\s*\d+)', block_content)
        cost_foreign = cost_foreign_match.group(1).replace(' ', '') if cost_foreign_match else "N/A"
            
        # --- Проходные баллы (История) ---
        scores_match = re.search(
            r'Проходные баллы:\s*(?:2025|—)\s*(?:2024|—)\s*(?:2023|—)\s*(?:2022|—)\s*(?:2021|—)\s*((\d+|—)\s+(\d+|—)\s+(\d+|—)\s+(\d+|—)\s+(\d+|—))',
            block_content, re.DOTALL | re.IGNORECASE
        )
        
        scores = ["N/A"] * 5
        if scores_match:
            score_values = re.findall(r'(\d+|—)', scores_match.group(1))
            if len(score_values) >= 5:
                # 2025 -> index 0, 2021 -> index 4
                scores = [s if s != '—' else "N/A" for s in score_values[:5]]


        # --- Количество мест ---
        quota_match = re.search(
            r'Бюджетные \(с учетом квот\)\s*Платные для граждан РФ\s*Платные для иностранных граждан\s*((\d+|—)\s+(\d+|—)\s+(\d+|—))', 
            block_content, re.DOTALL
        )

        places = ["0"] * 3
        if quota_match:
            place_values = re.findall(r'(\d+|—)', quota_match.group(1))
            if len(place_values) >= 3:
                places = [p if p != '—' else "0" for p in place_values[:3]]
        
        # --- Квоты ---
        separate_quota_match = re.search(r'Отдельная квота:\s*(\d+|—)\s*мест', block_content)
        special_quota_match = re.search(r'Особая квота:\s*(\d+|—)\s*мест', block_content)
        target_quota_match = re.search(r'Целевая квота:\s*(\d+|—)\s*мест', block_content)
        
        quota_separate = separate_quota_match.group(1) if separate_quota_match and separate_quota_match.group(1) != '—' else "0"
        quota_special = special_quota_match.group(1) if special_quota_match and special_quota_match.group(1) != '—' else "0"
        quota_target = target_quota_match.group(1) if target_quota_match and target_quota_match.group(1) != '—' else "0"

        
        # Сборка объекта
        program = {
            "Код": program_code,
            "Направление": program_name.strip(),
            "Форма_обучения": study_form,
            "Предметы": subjects_final,
            "Стоимость_РФ": cost_rf,
            "Стоимость_Иностр": cost_foreign,
            
            "Проходной_2025": scores[0],
            "Проходной_2024": scores[1],
            "Проходной_2023": scores[2],
            "Проходной_2022": scores[3],
            "Проходной_2021": scores[4],
            
            "Места_Бюджет": places[0],
            "Места_Платные_РФ": places[1],
            "Места_Платные_Иностр": places[2],
            
            "Квота_Отдельная": quota_separate,
            "Квота_Особая": quota_special,
            "Квота_Целевая": quota_target,
        }
        
        program["Уровень"] = get_program_level(program_code)
        
        structured_programs.append(program)
        
    return structured_programs

# =========================================================
# 4. ЗАПУСК
# =========================================================

def parse_stankin_page(url: str):
    """Основная точка входа."""
    html_code = fetch_html_content(url)

    if not html_code:
        logging.error("Не удалось получить контент страницы.")
        return

    pure_text = clean_html_content(html_code)
    structured_data = extract_structured_data(pure_text)
    
    print("\n" + "="*70)
    print(f"🤖 РЕЗУЛЬТАТ ПАРСИНГА: {url}")
    print("="*70)
    
    print(json.dumps(structured_data, ensure_ascii=False, indent=4))
    print(f"\nИТОГО: Успешно извлечено {len(structured_data)} программ.")

if __name__ == "__main__":
    test_url = "https://priem.stankin.ru/bakalavriatispetsialitet/training_programs/"
    parse_stankin_page(test_url)