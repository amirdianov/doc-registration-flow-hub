import datetime
from pathlib import Path

import spacy
import xml.etree.ElementTree as ET
from natasha import (
    Doc, Segmenter, NewsNERTagger, NewsEmbedding,
    NamesExtractor, MorphVocab, NewsMorphTagger,
    NewsSyntaxParser, DatesExtractor, MoneyExtractor, AddrExtractor
)
from yargy import Parser

from ml.ner.natasha_rules import R_SudD, R_INN, R_SNILS

# Категории сущностей
PERSON_SET = set()
LOCATION_SET = set()
ORGANIZATION_SET = set()
DATE_SET = set()
MONEY_SET = set()
ADDRESS_SET = set()
SUD_DELO_SET = set()
INN = set()
SNILS = set()


def spacy_entity(text: str):
    nlp = spacy.load("ru_core_news_md")
    # Обрабатываем текст spaCy
    doc_spacy = nlp(text)
    # Обрабатываем spaCy
    for ent in doc_spacy.ents:
        if ent.label_ == "PER" and len(ent.text.split()) == 3:
            PERSON_SET.add(ent.text)
        elif ent.label_ == "LOC":
            LOCATION_SET.add(ent.text)
        elif ent.label_ == "ORG":
            ORGANIZATION_SET.add(ent.text)


def natasha_entity(text):
    # Инициализация компонентов Natasha для обработки текста

    # 1. Инициализация основных компонентов обработки
    segmenter = Segmenter()  # Сегментатор текста на предложения
    morph_vocab = MorphVocab()  # Морфологический словарь для нормализации слов
    emb = NewsEmbedding()  # Предобученные векторные представления слов (эмбеддинги)

    # 2. Инициализация моделей для различных видов анализа
    morph_tagger = NewsMorphTagger(emb)  # Модель для морфологического разбора
    syntax_parser = NewsSyntaxParser(emb)  # Модель для синтаксического анализа
    ner_tagger = NewsNERTagger(emb)  # Модель для распознавания именованных сущностей (NER)

    # 3. Инициализация экстракторов для конкретных типов информации
    names_extractor = NamesExtractor(morph_vocab)  # Извлечение имен
    dates_extractor = DatesExtractor(morph_vocab)  # Извлечение дат
    money_extractor = MoneyExtractor(morph_vocab)  # Извлечение денежных сумм
    address_extractor = AddrExtractor(morph_vocab)  # Извлечение адресов

    # Обработка текста с помощью Natasha
    doc_natasha = Doc(text)  # Создание документа для обработки

    # Последовательное применение различных видов анализа:
    doc_natasha.segment(segmenter)  # Сегментация текста на предложения
    doc_natasha.tag_morph(morph_tagger)  # Морфологический разбор (часть речи, род, число и т.д.)
    doc_natasha.parse_syntax(syntax_parser)  # Синтаксический анализ (зависимости между словами)
    doc_natasha.tag_ner(ner_tagger)  # Распознавание именованных сущностей

    # Собираем данные из Natasha
    for ent in doc_natasha.spans:
        if ent.type == "PER" and len(ent.text.split()) == 3:
            PERSON_SET.add(ent.text)
        elif ent.type == "LOC":
            LOCATION_SET.add(ent.text)
        elif ent.type == "ORG":
            ORGANIZATION_SET.add(ent.text)

    # Извлечение имён Natasha
    for sentence in doc_natasha.sents:
        for match in names_extractor(sentence.text):
            name = match.fact
            if name.first and name.last and name.middle:
                PERSON_SET.add(f"{name.first} {name.last} {name.middle}")

    # Извлечение дат Natasha
    for sentence in doc_natasha.sents:
        for match in dates_extractor(sentence.text):
            date = match.fact
            if date.year and date.month and date.day:
                date_str = datetime.date(date.year, date.month, date.day).strftime("%d.%m.%Y")
                DATE_SET.add(date_str)

    # Извлечение сумм Natasha
    for sentence in doc_natasha.sents:
        for match in money_extractor(sentence.text):
            money = match.fact
            if money.amount:
                MONEY_SET.add(str(money.amount))

    # Извлечение адресов Natasha
    for sentence in doc_natasha.sents:
        for match in address_extractor(sentence.text):
            addr = match.fact
            if addr.value and addr.type:
                ADDRESS_SET.add(f"{addr.value} {addr.type}")

    parserSudD = Parser(R_SudD)
    parserINN = Parser(R_INN)
    parserSNILS = Parser(R_SNILS)

    for match in parserSudD.findall(text):
        countNUM = [x.value for x in match.tokens]
        if countNUM != 0:
            numberSudDelo = ''
            for i in range(len(countNUM)):
                if any(map(str.isdigit, countNUM[i])) or countNUM[i] in {'M', 'm', 'м', 'М', 'А', 'A', 'a', 'а'}:
                    startId = i
                    break
            for i in range(startId, len(countNUM)):
                numberSudDelo += countNUM[i]
            paramList = []
            paramList.append(numberSudDelo)
            SUD_DELO_SET.add(''.join(paramList))

    for match in parserINN.findall(text):
        countNUM = [x.value for x in match.tokens]
        if countNUM != 0:
            if countNUM[1] in [':']:
                numberINN = countNUM[2]
            else:
                numberINN = countNUM[1]
            paramList = []
            paramList.append(numberINN)
            INN.add(''.join(paramList))

    for match in parserSNILS.findall(text):
        countNUM = [x.value for x in match.tokens]
        if countNUM != 0:
            numberSNILS = countNUM[1]
            paramList = []
            paramList.append(numberSNILS)
            SNILS.add(''.join(paramList))


def make_xml():
    # Генерация XML
    root = ET.Element("Entities")

    # Функция для добавления сущностей
    def add_entities_to_xml(entity_set, label):
        # Группировка по категориям
        category_elem = ET.SubElement(root, "Category", type=label)
        for value in entity_set:
            ET.SubElement(category_elem, "Entity").text = value

    # Добавляем сущности
    add_entities_to_xml(PERSON_SET, "PER")
    add_entities_to_xml(LOCATION_SET, "LOC")
    add_entities_to_xml(ORGANIZATION_SET, "ORG")
    add_entities_to_xml(DATE_SET, "DATE")
    add_entities_to_xml(SUD_DELO_SET, "DELO")
    add_entities_to_xml(MONEY_SET, "MONEY")
    add_entities_to_xml(ADDRESS_SET, "ADDRESS")
    add_entities_to_xml(INN, "INN")
    add_entities_to_xml(SNILS, "SNILS")

    # Очищаем множества
    PERSON_SET.clear()
    LOCATION_SET.clear()
    ORGANIZATION_SET.clear()
    DATE_SET.clear()
    SUD_DELO_SET.clear()
    MONEY_SET.clear()
    ADDRESS_SET.clear()
    INN.clear()
    SNILS.clear()

    return root


def save_xml(root, file_path='', filename="entities.xml"):
    # Сохраняем XML
    xml_filename = Path(file_path).joinpath(filename)
    tree = ET.ElementTree(root)
    tree.write(xml_filename, encoding="utf-8", xml_declaration=True)


if __name__ == '__main__':
    with open('test.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    spacy_entity(text)
    natasha_entity(text)
    root = make_xml()
    save_xml(root)
