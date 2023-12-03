import cv2
import pytesseract
import easyocr
import pathlib
import csv
import warnings
from difflib import SequenceMatcher

# import enchant
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    reader_easy = easyocr.Reader(['en', 'ru'])

# dictionary = enchant.Dict("ru_RU")
pytesseract.pytesseract.tesseract_cmd = (r"C:/Program Files/Tesseract-OCR/tesseract.exe")


# os.environ['EASYOCR_LOGGER'] = 'ERROR'
# reader = easyocr.Reader(['en', 'ru'])

def rel_path(rel_path):
    path = pathlib.Path(__file__).parent / rel_path
    return path


# Используем EasyOCR.
def easyocr_recognition(image_paths):
    predictions = {}
    for image_path in image_paths:
        result = reader_easy.readtext(image_path)
        text = ' '.join([item[1] for item in result])
        predictions[image_path.split('/')[-1]] = text.strip()
    return predictions


def tesseract_recognition(image_paths):
    predictions = {}
    for image_path in image_paths:
        img = cv2.imread(image_path)
        result = pytesseract.image_to_string(img, lang='rus+eng')
        predictions[image_path.split('/')[-1]] = result.strip()
    return predictions

def boxes_recognition(image_paths):
    predictions = {}
    for image_path in image_paths:
        img = cv2.imread(image_path)
        h, w, _ = img.shape
        boxes = pytesseract.image_to_boxes(img, lang='rus+eng')
        result = ''.join([sym_data.split(' ')[0] for sym_data in boxes.split('\n')])
        predictions[image_path.split('/')[-1]] = result.strip()
    return predictions


def check_binary_correct(predictions, label):
    output_str = ''
    count_correct = 0
    image_count = len(predictions.keys())
    for file in predictions.keys():
        output_str += f'{file} | {label[file].lower()} | {predictions[file].lower()}\n'
        if predictions[file].lower() == label[file].lower():
            count_correct += 1
    output_str += f'Статистика: угадано {count_correct} / {image_count} капч'
    return output_str


def check_sequence_matcher(predictions, label):
    output_str = ''
    similarities = []
    for file in predictions.keys():
        similarity = SequenceMatcher(None, label[file].lower(), predictions[file].lower()).ratio()
        similarities.append(similarity)
        output_str += f'{file} | {label[file].lower()} | {predictions[file].lower()}\n'
    output_str += f'Статистика: средняя схожесть: {statistics.fmean(similarities) * 100}%'
    return output_str


def test_recognition(rec_type, val_type, dataset_num):
        output_str = ''
        labels = {}
        images_count = 0
        correct_guesses = 0
        similarities = []
        if dataset_num ==2:
            with open(str(rel_path('dataset_aug/labels.csv')), newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile, delimiter=',', quotechar="'")
                for row in reader:
                    labels[row[0]] = row[1]

            img_files=[]
            for x in range(11):
                for rotate in range(21):
                    img_files.append(f'dataset_aug/samples/{x+1}_rotated_{rotate}.jpg')
                    if rotate!=0:
                        img_files.append(f'dataset_aug/samples/{x+1}_rotated_-{rotate}.jpg')
        else:
            with open(str(rel_path('dataset/labels.csv')), newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile, delimiter=',', quotechar="'")
                for row in reader:
                    labels[row[0]] = row[1]
            img_files = [f'dataset/samples/{x + 1}.jpg' for x in range(11)]


        if rec_type == 'straight_recognition':
            predictions = tesseract_recognition(img_files)
        if rec_type == 'boxes_recognition':
            predictions = boxes_recognition(img_files)

        # elif rec_type == 'easyocr':
        #     predictions = easyocr_recognition(img_files)
        #
        #
        # if val_type == 'binary_correct':
        #     output_str = check_binary_correct(predictions, labels)
        #
        # if val_type == 'similarity':
        #     output_str = check_sequence_matcher(predictions, labels)
        #
        # with open(str(rel_path('results.txt')), 'w', encoding='utf-8') as f:
        #     f.write(output_str)


test_recognition('boxes_recognition', 'binary_correct', 1)
