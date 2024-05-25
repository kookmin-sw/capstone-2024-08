""" from https://github.com/keithito/tacotron """
from . import cleaners
from . import symbols


# # Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}


def text_to_sequence(text, cleaner_names):
    '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
      Args:
        text: string to convert to a sequence
        cleaner_names: names of the cleaner functions to run the text through
      Returns:
        List of integers corresponding to the symbols in the text
    '''
    sequence = []

    clean_text = _clean_text(text, cleaner_names)
    for symbol in clean_text:
        symbol_id = _symbol_to_id[symbol]
        sequence += [symbol_id]
    return sequence


def cleaned_text_to_sequence(cleaned_text):
    '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
      Args:
        text: string to convert to a sequence
      Returns:
        List of integers corresponding to the symbols in the text
    '''
    sequence = [_symbol_to_id[symbol] for symbol in cleaned_text]
    return sequence


def sequence_to_text(sequence):
    '''Converts a sequence of IDs back to a string'''
    result = ''
    for symbol_id in sequence:
        s = _id_to_symbol[symbol_id]
        result += s
    return result


def _clean_text(text, cleaner_names):
    """
    다양한 클리너를 사용한다면 주석 처리된 코드 사용
    현재는 한국어 클리너만 사용
    """
    # for name in cleaner_names:
    #     cleaner = getattr(cleaners, name)
    #     if not cleaner:
    #         raise Exception('Unknown cleaner: %s' % name)
    text = cleaners.korean_cleaners(text)
    return text
