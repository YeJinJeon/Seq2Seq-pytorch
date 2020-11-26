import helper
import math
from preprocess import *
from model import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_demo_batches(sources, batch_size, source_pad_int, target_pad_int, trg_int_eos):
    """Batch targets, sources, and the lengths of their sentences together"""
    for batch_i in range(0, len(sources) // batch_size):
        start_i = batch_i * batch_size
        sources_batch = sources[start_i:start_i + batch_size]
        # targets_batch = targets[start_i:start_i + batch_size]
        pad_sources_batch = np.array(pad_sentence_batch(sources_batch, source_pad_int))
        # pad_targets_batch = np.array(pad_sentence_batch(targets_batch, target_pad_int))
        pad_targets_batch = np.zeros_like(pad_sources_batch) ########################################
        trg_int_eos = np.asarray(trg_int_eos).reshape(1, -1) ########################################
        pad_targets_batch = np.append(pad_targets_batch, trg_int_eos, axis=1) ###############################
        # Need the lengths for the _lengths parameters
        pad_targets_lengths = []
        for target in pad_targets_batch:
            pad_targets_lengths.append(len(target))

        pad_source_lengths = []
        for source in pad_sources_batch:
            pad_source_lengths.append(len(source))

        yield pad_targets_batch, pad_sources_batch, pad_targets_lengths, pad_source_lengths


def demo(model, criterion, demo_source, batch_size, \
         src_int_pad, trg_int_pad, trg_int_go, trg_int_eos, src_int_to_letter, trg_int_to_letter):

    model.eval()
    epoch_loss = 0
    batch_num = 0

    with torch.no_grad():

        for i, (targets_batch, sources_batch, targets_lengths, sources_lengths) \
                in enumerate(get_demo_batches(demo_source, batch_size, src_int_pad, trg_int_pad, trg_int_eos)):

            targets_batch = process_decoder_input(targets_batch, trg_int_go, batch_size)

            # input [str_len, batch_size]
            src = torch.LongTensor(sources_batch).transpose(0, 1)
            trg = torch.LongTensor(targets_batch).transpose(0, 1)

            # print("******************")
            # print('  Input Words: {}'.format(" ".join([src_int_to_letter[i.item()] for i in src])))
            # print('  Response Words: {}'.format(" ".join([trg_int_to_letter[i.item()] for i in trg])))

            if src.shape[0] == 0:
                continue

            output = model(src, trg, 0) #*(trg_len, batch, output_dim)

            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim) #remove <GO>, shape:[out_len*batch_size, output_dim]
            trg = trg[1:].reshape(-1) #remove <GO>, shape: [trg len * batchsize]

            loss = criterion(output, trg)
            # print(loss)
            epoch_loss += loss.item()
            batch_num += 1

            predict = output.argmax(1)
            # pad = src_letter_to_int["<PAD>"]
            pad = src_int_pad

            print("************************")
            print('  Input Ids:       {}'.format([i for i in src]))
            print('  Input Words: {}'.format(" ".join([src_int_to_letter[i.item()] for i in src])))
            # print('  Target Ids:       {}'.format([i for i in trg]))
            # print('  Target Words: {}'.format(" ".join([trg_int_to_letter[i.item()] for i in trg])))
            print('  Response Ids:       {}'.format([i for i in predict if i != pad]))
            print('  Response Words: {}'.format(" ".join([trg_int_to_letter[i.item()] for i in predict if i != pad])))

        return epoch_loss / batch_num

if __name__ == '__main__':

    batch_size = 1
    source_path = 'data/custom_source.txt'
    target_path = 'data/custom_target.txt'

    source_sentences = helper.load_data(source_path)
    target_sentences = helper.load_data(target_path)

    # build int2letter and letter2int dicts
    source_int_to_letter, source_letter_to_int = extract_character_vocab(source_sentences)
    target_int_to_letter, target_letter_to_int = extract_character_vocab(target_sentences)

    # Convert characters to ids ex)bsaqq --> [[28,29,11,13,13],[]]
    source_letter_ids = [[source_letter_to_int.get(letter, source_letter_to_int['<UNK>']) for letter in line] for line
                         in source_sentences.split('\n')]
    # target_letter_ids = [[target_letter_to_int.get(letter, target_letter_to_int['<UNK>']) for letter in line] + [
    #     target_letter_to_int['<EOS>']] for line in target_sentences.split('\n')]

    # input_dim = len(source_letter_ids)
    # output_dim = len(target_letter_ids)
    input_dim = len(source_letter_to_int)
    output_dim = len(target_letter_to_int)
    enc_dim = 15
    dec_dim = 15
    hid_dim = 50
    n_layers = 2
    n_epochs = 60
    learning_rate = 0.001

    enc = Encoder(input_dim, enc_dim, hid_dim, n_layers)
    print(enc)
    dec = Decoder(output_dim, dec_dim, hid_dim, n_layers)
    print(dec)
    model = Seq2Seq(enc, dec, device).to(device)
    criterion = nn.CrossEntropyLoss()

    model.load_state_dict(torch.load('checkpoints_dropout/rnn-tut1-model.pt'))
    # model.load_state_dict(torch.load('checkpoints/epoch_160_loss_0.099143.pt'))

    test_loss = demo(model, criterion, source_letter_ids, batch_size, source_letter_to_int['<PAD>'],
                        target_letter_to_int['<PAD>'], target_letter_to_int['<GO>'], target_letter_to_int['<EOS>'], source_int_to_letter, target_int_to_letter)

    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')