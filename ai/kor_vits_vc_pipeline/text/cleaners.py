# coding: utf-8

# Code based on https://github.com/keithito/tacotron/blob/master/text/cleaners.py
'''
Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
    1. "english_cleaners" for English text
    2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
         the Unidecode library (https://pypi.python.org/pypi/Unidecode)
    3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
         the symbols in symbols.py to match your data).
'''

import re
from .korean import tokenize as ko_tokenize
from phonemizer import phonemize
import jamotools

# Added to support LJ_speech
from unidecode import unidecode
from .en_numbers import normalize_numbers as en_normalize_numbers

import g2pk
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


def english_cleaners(text):
    '''Pipeline for English text, including abbreviation expansion.'''
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_abbreviations(text)
    phonemes = phonemize(text, language='en-us', backend='espeak', strip=True)
    phonemes = collapse_whitespace(phonemes)
    return phonemes


def english_cleaners2(text):
    '''Pipeline for English text, including abbreviation expansion. + punctuation + stress'''
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_abbreviations(text)
    phonemes = phonemize(text, language='en-us', backend='espeak', strip=True, preserve_punctuation=True,
                         with_stress=True)
    phonemes = collapse_whitespace(phonemes)
    return phonemes


def korean_cleaners(text):
    '''Pipeline for Korean text, including number and abbreviation expansion.'''
    global g2p_dict, phoneme

    text = jamotools.join_jamos(text)

    try:
        phoneme = g2p_dict[text]
    except KeyError:
        phoneme = g2p(text, descriptive=True, group_vowels=True)
        g2p_dict[text] = phoneme
    finally:
        text = phoneme

    text = jamotools.split_syllables(text, jamo_type="JAMO")
    text = text.replace('@', '')

    phonemes = phonemize(text, language='ko', backend='espeak', strip=True, preserve_punctuation=True,
                         with_stress=True)
    phonemes = collapse_whitespace(phonemes)
    phonemes = phonemes.replace('(en)', '')
    phonemes = phonemes.replace('(ko)', '')
    phonemes = phonemes.replace('-', '')

    return phonemes
