# coding: utf-8

# Code based on https://github.com/keithito/tacotron/blob/master/text/cleaners.py


import re
from .korean import tokenize as ko_tokenize
from phonemizer import phonemize
import jamotools

from unidecode import unidecode
# 영어를 위한 전처리
# from en_numbers import normalize_numbers as en_normalize_numbers

# import g2pk
from g2pk import G2p

from kiwipiepy import Kiwi

g2p = G2p()
g2p_dict = dict()
kiwi = Kiwi()

# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
    ('mrs', 'misess'),
    ('mr', 'mister'),
    ('dr', 'doctor'),
    ('st', 'saint'),
    ('co', 'company'),
    ('jr', 'junior'),
    ('maj', 'major'),
    ('gen', 'general'),
    ('drs', 'doctors'),
    ('rev', 'reverend'),
    ('lt', 'lieutenant'),
    ('hon', 'honorable'),
    ('sgt', 'sergeant'),
    ('capt', 'captain'),
    ('esq', 'esquire'),
    ('ltd', 'limited'),
    ('col', 'colonel'),
    ('ft', 'fort'),
]]


def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def expand_numbers(text):
    return normalize_numbers(text)


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_re, ' ', text)


def convert_to_ascii(text):
    return unidecode(text)


def basic_cleaners(text):
    '''Basic pipeline that lowercases and collapses whitespace without transliteration.'''
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def transliteration_cleaners(text):
    '''Pipeline for non-English text that transliterates to ASCII.'''
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def korean_cleaners(text):
    '''Pipeline for Korean text, including number and abbreviation expansion.'''
    global g2p_dict

    text = jamotools.join_jamos(text)

    try:
        phoneme = g2p_dict.get(text)
        if phoneme is None:
            phoneme = g2p(text, descriptive=True, group_vowels=True)
            g2p_dict[text] = phoneme
    except Exception as e:
        print("Error during g2p conversion:", e)
        phoneme = text  # Fallback to original text if g2p conversion fails

    text = jamotools.split_syllables(phoneme, jamo_type="JAMO")
    text = text.replace('@', '')

    phonemes = phonemize(text, language='ko', backend='espeak',
                         strip=True, preserve_punctuation=True, with_stress=True)
    phonemes = collapse_whitespace(phonemes)
    phonemes = phonemes.replace('(en)', '').replace(
        '(ko)', '').replace('-', '')

    return phonemes
