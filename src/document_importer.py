import pathlib
import textract
import os
import re

os.environ["PATH"] += os.pathsep + "C:\\Program Files\\Git\\mingw64\\bin"


SUPPORTED_TEXT_TYPES = [
    '.doc'
]


def import_doc(input_path: pathlib.Path):
    doc_text = textract.process(input_path)
    encoding_type = 'utf-8'
    return doc_text.decode(encoding_type)


def import_document(input_path: pathlib.Path):
    doc_text = ''
    if input_path.suffix == '.doc':
        doc_text = import_doc(str(input_path))
    return doc_text


def preprocess(content: str):
    content = content[content.find('MR'):]
    content = content.replace('\n', ' ').replace('\r', ' ')
    content = ''.join((
        char
        for char in content
        if char.isalpha() or char == ' '
    ))
    content = re.sub(r"\s\s+", " ", content)
    content = content.strip()
    return content


def import_directory(input_path: pathlib.Path):
    result = []
    for file in input_path.iterdir():
        if file.suffix in SUPPORTED_TEXT_TYPES:
            file_content = import_document(file)
            file_content = preprocess(file_content)
            result.append(file_content)
    return result
