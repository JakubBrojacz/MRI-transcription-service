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
    content = content.lower()
    if 'mr' in content:
        content = content[content.find('mr'):]
    content = content.replace('\n', ' ').replace('\r', ' ')
    content = ''.join((
        char
        for char in content
        if char.isalpha() or char == ' '
    ))
    content = re.sub(r"\s\s+", " ", content)
    content = content.strip()
    return content


def import_file_list(file_list):
    result = []
    for file in file_list:
        if file.suffix in SUPPORTED_TEXT_TYPES:
            print(file.name)
            try:
                file_content = import_document(file)
                # file_content = file_content.lower()
                # if 'mr' in file_content:
                #     file_content = file_content[file_content.find('mr'):]
                # for f_c in file_content.split('.'):
                #     result.append(preprocess(f_c))
                file_content = preprocess(file_content)
                result.append(file_content)
            except Exception:
                print(f"ERROR with document {file}")
    return result


def import_directory(input_path: pathlib.Path):
    return import_file_list(input_path.rglob("*"))
    
