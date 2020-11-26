import helper
import math
from preprocess import *
from model import *
from nltk.metrics.distance import edit_distance

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test(model, criterion, test_target, test_source, batch_size, \
         src_int_pad, trg_int_pad, trg_int_go, src_int_to_letter, trg_int_to_letter):

    model.eval()
    epoch_loss = 0
    batch_num = 0
    evaluate_num = 0

    n_correct = 0
    norm_ED = 0
    n_correct_before = 0
    norm_ED_before = 0

    f = open("RNN_result.txt", 'w')
    with torch.no_grad():

        for i, (targets_batch, sources_batch, targets_lengths, sources_lengths) \
                in enumerate(get_batches(test_target, test_source, batch_size, src_int_pad, trg_int_pad)):

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
            pad = trg_int_pad

            #print("************************")
            #print('  Input Ids:       {}'.format([i for i in src]))
            #print('  Input Words: {}'.format(" ".join([src_int_to_letter[i.item()] for i in src])))
            #print('  Target Ids:       {}'.format([i for i in trg]))
            #print('  Target Words: {}'.format(" ".join([trg_int_to_letter[i.item()] for i in trg])))
            #print('  Response Ids:       {}'.format([i for i in predict if i != pad]))
            #print('  Response Words: {}'.format(" ".join([trg_int_to_letter[i.item()] for i in predict if i != pad])))

            source_str = "".join([src_int_to_letter[i.item()] for i in src])
            target_str = "".join([trg_int_to_letter[i.item()] for i in trg])
            predict_str = "".join([trg_int_to_letter[i.item()] for i in predict if i != pad])
            f.write("********************************\n")
            f.write("Input Words: " + source_str + "\n")
            f.write("Target Words: " + target_str + "\n")
            f.write("Response Words: " + predict_str + "\n")

            if len(predict_str) >= 5:
                evaluate_num += 1
                if predict_str == target_str:  # exactly same
                    n_correct += 1
                # ICDAR2019 Normalized Edit Distance
                if len(target_str) == 0 or len(predict_str) == 0:
                    norm_ED += 0
                elif len(target_str) > len(predict_str):
                    norm_ED += 1 - edit_distance(predict_str, target_str) / len(target_str)
                else:
                    norm_ED += 1 - edit_distance(predict_str, target_str) / len(predict_str)

                if source_str == target_str:  # exactly same
                    n_correct_before += 1
                # ICDAR2019 Normalized Edit Distance
                if len(target_str) == 0 or len(source_str) == 0:
                    norm_ED_before += 0
                elif len(target_str) > len(source_str):
                    norm_ED_before += 1 - edit_distance(source_str, target_str) / len(target_str)
                else:
                    norm_ED_before += 1 - edit_distance(source_str, target_str) / len(source_str)

        accuracy = n_correct / float(evaluate_num) * 100
        norm_ED = norm_ED / float(evaluate_num)  # ICDAR2019 Normalized Edit Distance

        accuracy_before = n_correct_before / float(evaluate_num) * 100
        norm_ED_before = norm_ED_before / float(evaluate_num)  # ICDAR2019 Normalized Edit Distance

        print("Accuracy, norm_ED:")
        print(accuracy_before, norm_ED_before)
        print("*******************************************")
        print("Accuracy, norm_ED:")
        print(accuracy, norm_ED)
        f.close()
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
    target_letter_ids = [[target_letter_to_int.get(letter, target_letter_to_int['<UNK>']) for letter in line] + [
        target_letter_to_int['<EOS>']] for line in target_sentences.split('\n')]

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

    model.load_state_dict(torch.load('checkpoints/rnn-tut1-model.pt'))
    # model.load_state_dict(torch.load('checkpoints/epoch_160_loss_0.099143.pt'))

    test_loss = test(model, criterion, target_letter_ids , source_letter_ids, batch_size, source_letter_to_int['<PAD>'],
                        target_letter_to_int['<PAD>'], target_letter_to_int['<GO>'], source_int_to_letter, target_int_to_letter)

    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')