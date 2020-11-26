import numpy as np
import helper
import random

"""
deep spelling - pytorch version
reference:https://github.com/gaushh/Deep-Spelling/blob/master/deep_spell_GRU_attention_train.ipynb
"""
def ordered_unique_list(input_list):
    input_dic = {}
    r_list = []
    for i, v in enumerate(input_list):
        get_value = input_dic.get(v, None)
        if get_value == None:
            input_dic[v] = i
            r_list.append(v)
    return r_list

def extract_character_vocab(data):
    """
    :param data: contents in txt file
    :return int_to_vocab {0:<PAD>, 1:<UNK>, 2:<GO>, 3:<EOS>, 4:'o', 5:'r'..}
            vocab_t_int {'<PAD>':0, '<UNK>:1, '<GO>':2, '<EOS>':3, 'o':4, 'r':5..}
    """
    special_words = ['<PAD>', '<UNK>', '<GO>',  '<EOS>']

    # set_words = set([character for line in data.split('\n') for character in line])
    # set_words = sorted(set([character for line in data.split('\n') for character in line]))
    set_words = ordered_unique_list([character for line in data.split('\n') for character in line])
    random.seed(4)
    random.shuffle(set_words)
    int_to_vocab = {word_i: word for word_i, word in enumerate(special_words + list(set_words))}
    vocab_to_int = {word: word_i for word_i, word in int_to_vocab.items()}

    return int_to_vocab, vocab_to_int

def pad_sentence_batch(sentence_batch, pad_int):
    """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]

def get_batches(targets, sources, batch_size, source_pad_int, target_pad_int):
    """Batch targets, sources, and the lengths of their sentences together"""
    for batch_i in range(0, len(sources) // batch_size):
        start_i = batch_i * batch_size
        sources_batch = sources[start_i:start_i + batch_size]
        targets_batch = targets[start_i:start_i + batch_size]
        pad_sources_batch = np.array(pad_sentence_batch(sources_batch, source_pad_int))
        pad_targets_batch = np.array(pad_sentence_batch(targets_batch, target_pad_int))

        # Need the lengths for the _lengths parameters
        pad_targets_lengths = []
        for target in pad_targets_batch:
            pad_targets_lengths.append(len(target))

        pad_source_lengths = []
        for source in pad_sources_batch:
            pad_source_lengths.append(len(source))

        yield pad_targets_batch, pad_sources_batch, pad_targets_lengths, pad_source_lengths

def process_decoder_input(target_data, vocab_to_int_GO, batch_size):
    '''Remove the last word id from each batch and concat the <GO> to the begining of each batch'''
    # ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    # dec_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1)

    ending = target_data[:, :-1]
    dec_input = np.insert(ending, 0, vocab_to_int_GO, axis=1)

    return dec_input

if __name__ == "__main__":
    batch_size = 2
    source_path = 'data/letters_source.txt'
    target_path = 'data/letters_target.txt'

    source_sentences = helper.load_data(source_path)
    target_sentences = helper.load_data(target_path)

    # build int2letter and letter2int dicts
    source_int_to_letter, source_letter_to_int = extract_character_vocab(source_sentences)
    target_int_to_letter, target_letter_to_int = extract_character_vocab(target_sentences)

    # Convert characters to ids ex)bsaqq --> [[28,29,11,13,13],[]]
    source_letter_ids = [[source_letter_to_int.get(letter, source_letter_to_int['<UNK>']) for letter in line] for line
                         in source_sentences.split('\n')]
    target_letter_ids = [[target_letter_to_int.get(letter, target_letter_to_int['<UNK>']) for letter in line] + [
        target_letter_to_int['<EOS>']] for line in target_sentences.split('\n')]

    (targets_batch, sources_batch, targets_lengths, sources_lengths) = next(get_batches(target_letter_ids, source_letter_ids, batch_size,
                source_letter_to_int['<PAD>'],
                target_letter_to_int['<PAD>']))

    dec_input = process_decoder_input(targets_batch, target_letter_to_int, batch_size)
    print()


