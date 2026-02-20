'''Скрипт для отладки парсера табличных данных'''

from table_parser import fetch_html_content, clean_html_content

url = "https://priem.stankin.ru/bakalavriatispetsialitet/training_programs/"
print(f"1. Качаем {url}...")
html = fetch_html_content(url)

if html:
    print("2. Чистим текст...")
    text = clean_html_content(html)
    
    with open("debug_table_parser.txt", "w", encoding="utf-8") as f:
        f.write(text)
        
    print("\n✅ Готово! Файл 'debug_table_parser.txt' создан.")
    print("="*60)
    print("ВОТ КУСОК ТЕКСТА ДЛЯ ПРИМЕРА (ПЕРВЫЕ 2000 символов):")
    print("="*60)
    
    # Найдем начало первой программы и выведем кусок в консоль
    start_index = text.find("09.03.01")
    if start_index != -1:
        print(text[start_index : start_index + 1500])
    else:
        print("Не нашел 09.03.01 в тексте. Вывожу начало файла:\n")
        print(text[:1000])