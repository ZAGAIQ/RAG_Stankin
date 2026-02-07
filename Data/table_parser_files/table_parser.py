import requests
import logging
import re
import json
from typing import Optional, List, Dict
from bs4 import BeautifulSoup, Comment
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# =========================================================
# 1. ЗАГРУЗКА И ОЧИСТКА (Твои функции, они ок)
# =========================================================

def fetch_html_content(url: str) -> Optional[str]:
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, timeout=15, headers=headers)
        response.encoding = response.apparent_encoding
        return response.text
    except Exception as e:
        logging.error(f"Ошибка: {e}")
        return None

def clean_html_content(html_content: str) -> str:
    soup = BeautifulSoup(html_content, 'lxml')
    
    for element in soup(['script', 'style', 'noscript', 'iframe', 'meta', 'link', 'br']):
        element.decompose()
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()
    
    # Удаляем меню и футеры
    for element in soup.find_all('div', class_=['block-0-menu-16', 'landing-footer']):
        element.decompose()
        
    main_content = soup.find('div', class_='landing-main') or soup.body
    
    # Добавляем пробелы, чтобы текст не склеился
    for tag in main_content.find_all(['h1', 'h2', 'h3', 'p', 'div', 'li', 'td', 'span']):
        tag.insert_after(' ')

    text = main_content.get_text(separator=' ', strip=True)
    return re.sub(r'\s+', ' ', text) # Превращаем всё в одну длинную строку

# =========================================================
# 2. НОВАЯ ЛОГИКА ПАРСИНГА (TOKEN BASED)
# =========================================================

def get_program_level(code: str) -> str:
    if ".03." in code: return "Бакалавриат"
    if ".05." in code: return "Специалитет"
    return "Магистратура/Другое"


def normalize_subjects(raw_subjects: str) -> List[str]:
    """
    Превращает строку 'Р + М + И/Ф' в список ['Русский язык', 'Математика', 'Информатика', 'Физика']
    """
    # Только те предметы, которые реально встречаются или могут встретиться
    mapping = {
        'Р': 'Русский',
        'М': 'Математика',
        'И': 'Информатика',
        'Ф': 'Физика',
        'Х': 'Химия',
        'О': 'Обществознание',
        'ИЯ': 'Иностранный',
        'Б': 'Биология'
    }
    
    # 1. Удаляем скобки с баллами и "min"
    clean_str = re.sub(r'\(.*?\)', '', raw_subjects)
    
    # 2. Разбиваем по разделителям (+ или /) и чистим пробелы
    tokens = re.split(r'\s*[+/]\s*', clean_str)
    
    final_list = []
    for t in tokens:
        key = t.strip().upper()
        if not key: continue
        
        if key in mapping:
            final_list.append(mapping[key])
        else:
            # Если встретился 'X' или что-то неизвестное, оставляем как есть
            final_list.append(key)
            
    return sorted(list(set(final_list)))


def extract_structured_data(full_text: str) -> List[Dict]:
    """
    Парсинг v5.0: Логика токенов. Мы не ищем 'число после слова', 
    мы берем все числа в блоке и расставляем их по порядку.
    """
    # Разбиваем текст на блоки по кодам программ (XX.XX.XX)
    # (?<!\d) - проверка, что перед кодом нет цифры (чтобы не разорвать год или цену)
    split_pattern = r'(?<!\d)(\d{2}\.\d{2}\.\d{2}(?:\.\d{2})?)'
    parts = re.split(split_pattern, full_text)
    
    programs = []
    
    # parts[0] - мусор. Дальше: parts[1]=Код, parts[2]=Текст, parts[3]=Код...
    for i in range(1, len(parts), 2):
        if i + 1 >= len(parts): break
        
        code = parts[i].strip()
        text = parts[i+1] # Весь текст описания программы
        
        # Если блок слишком короткий — пропускаем
        if len(text) < 50 or "Форма обучения" not in text:
            continue

        # --- 1. Название ---
        # Текст от начала до "Форма обучения"
        name_match = re.search(r'^\s*(.*?)(?=Форма обучения)', text)
        name = name_match.group(1).strip() if name_match else "N/A"
        name = re.sub(r'^[\s\.\-]+', '', name) # Чистим мусор в начале

        # --- 2. Форма обучения ---
        form_match = re.search(r'Форма обучения[:\s]*([а-яА-Я]+)', text)
        form = form_match.group(1) if form_match else "очная"

        # --- 3. Предметы ---
        subj_match = re.search(r'Предметы[:\s]*(.*?)(?=Количество мест)', text)
        subjects = subj_match.group(1).strip() if subj_match else "N/A"

        # Нормализация для метаданных
        subjects_list = normalize_subjects(subjects)

        # --- 4. Цены ---
        # Ищем все "руб" и берем числа перед ними
        prices = re.findall(r'(\d[\d\s]*)\s*руб', text)
        clean_prices = [p.replace(' ', '') for p in prices]
        cost_rf = clean_prices[0] if len(clean_prices) > 0 else "0"
        cost_foreign = clean_prices[1] if len(clean_prices) > 1 else "0"

        # --- 5. МЕСТА (Самое важное) ---
        # Изолируем кусок текста про места: от "Количество мест" до "Отдельная квота" (или до баллов)
        seats_block_match = re.search(r'Количество мест.*?(?=Отдельная квота|Проходные баллы)', text)
        budget, paid_rf, paid_foreign = "0", "0", "0"
        
        if seats_block_match:
            block = seats_block_match.group(0)
            
            # Находим ВСЕ "токены", похожие на кол-во мест (числа или прочерки)
            # Игнорируем длинные числа (годы 2024, 2025)
            # Паттерн: число из 1-3 цифр ИЛИ прочерк
            tokens = re.findall(r'(?<!\d)(\d{1,3}|—)(?!\d)', block)
            
            # Фильтруем, если вдруг попал год типа '25' (хотя вряд ли)
            # Берем последние 3 токена. Обычно порядок: [Бюджет, Платное РФ, Платное Иностр]
            if len(tokens) >= 3:
                # Берем с конца, так надежнее
                vals = tokens[-3:]
                budget = vals[0] if vals[0] != '—' else '0'
                paid_rf = vals[1] if vals[1] != '—' else '0'
                paid_foreign = vals[2] if vals[2] != '—' else '0'
            elif len(tokens) > 0:
                # Если нашли только одно число — это скорее всего бюджет
                budget = tokens[0] if tokens[0] != '—' else '0'

        # --- 6. Баллы ---
        # Удаляем цены из текста, чтобы не мешались
        text_no_prices = re.sub(r'\d[\d\s]*\s*руб', '', text)
        
        # Ищем баллы в диапазоне 110-310 (защита от года и цен)
        scores = re.findall(r'\b(\d{3})\b', text_no_prices)
        valid_scores = [s for s in scores if 110 <= int(s) <= 310]
        
        # Берем первые 5 найденных
        final_scores = (valid_scores + ["N/A"] * 5)[:5]

        programs.append({
            "Код": code,
            "Направление": name,
            "Форма": form,
            "Предметы": subjects,       # Оригинал (для текста)
            "Предметы_Список": subjects_list, # Список (для поиска/фильтров)
            "Бюджет": budget,
            "Платное_РФ": paid_rf,
            "Платное_Иностр": paid_foreign,
            "Стоимость_РФ": cost_rf,
            "Стоимость_Иностр": cost_foreign,
            # ИСПРАВЛЕННЫЙ ПОРЯДОК: Первое найденное число - это текущий год (2025)
            "Балл_2025": final_scores[0],
            "Балл_2024": final_scores[1],
            "Балл_2023": final_scores[2],
            "Балл_2022": final_scores[3],
            "Балл_2021": final_scores[4],
            "Уровень": get_program_level(code)
        })

    return programs

def save_to_json(data: List[Dict], filename: str):
    """Сохраняет данные в JSON файл."""
    # Создаем папку data, если её нет
    filepath = os.path.join(filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"💾 Данные успешно сохранены в: {filepath}")

# =========================================================
# 3. ЗАПУСК
# =========================================================

if __name__ == "__main__":
    url = "https://priem.stankin.ru/bakalavriatispetsialitet/training_programs/"
    print(f"Парсим: {url}")
    
    html = fetch_html_content(url)
    if html:
        print("1. HTML получен.")
        clean_text = clean_html_content(html)
        print("2. Текст очищен.")
        data = extract_structured_data(clean_text)
        print(f"3. Извлечено {len(data)} программ.")
        
        # Сохраняем результат
        save_to_json(data, "Data//table_parser_files//stankin_programs.json")