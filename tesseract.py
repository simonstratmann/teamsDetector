# -*- coding: utf-8 -*-
import Levenshtein
import pytesseract
from PIL import Image


def findMatch(known, text):
    found = [f.lower() for f in text.split() if len(f) >= 3]
    for k in known:
        distances = []
        for i in range(0, min(len(k), len(found))):
            if len(found[i]) >= 3:
                distances.append(Levenshtein.distance(k[i].lower(), found[i].lower()))
        if all(x <= 3 for x in distances):
            print(f"{k} matches {found}")


if __name__ == '__main__':
    pytesseract.pytesseract.tesseract_cmd = r'c:\Program Files\Tesseract-OCR\tesseract.exe'
    strings = []
    strings.append(pytesseract.image_to_string(Image.open('nameBoxJannik.png')))
    strings.append(pytesseract.image_to_string(Image.open('nameBoxJörn.png')))

    known = [["Jannik", "Schick"],
             [u"Jörn", "Hauke"],
             ["Hauke", "Plambeck"],
             ["Hauke", u"Schäfer"],
             ["Axel", "Miller"],
             ["Axel", "Dehning"],
             ]


    findMatch(known, "jom jauke *")
    findMatch(known, "jannik schik")
    findMatch(known, "Axl Milller")

