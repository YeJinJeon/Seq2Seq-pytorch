import numpy as np
import helper
import os
import time
import math
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from preprocess import *
from model import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, optimizer, criterion, train_target, train_source, batch_size, src_int_pad, trg_int_pad, trg_int_go):
    model.train()
    epoch_loss = 0
    batch_num = 0
    for i, (targets_batch, sources_batch, targets_lengths, sources_lengths) \
            in enumerate(get_batches(train_target, train_source, batch_size, src_int_pad, trg_int_pad)):

        targets_batch = process_decoder_input(targets_batch, trg_int_go, batch_size)

        # input [str_len, batch_size]
        src = torch.LongTensor(sources_batch).transpose(0, 1)
        trg = torch.LongTensor(targets_batch).transpose(0, 1)
        optimizer.zero_grad()
        output = model(src, trg)

        output_dim = output.shape[-1]

        output = output[1:].view(-1, output_dim) #remove <GO>, shape:[out_len*batch_size, output_dim]
        trg = trg[1:].reshape(-1) #remove <GO>, shape: [trg len * batchsize]

        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        batch_num += 1
    return epoch_loss / batch_num

def evaluate(model, criterion, valid_target, valid_source, batch_size, src_int_pad, trg_int_pad, trg_int_go):
    model.eval()
    epoch_loss = 0
    batch_num = 0

    with torch.no_grad():

        for i, (targets_batch, sources_batch, targets_lengths, sources_lengths) \
                in enumerate(get_batches(valid_target, valid_source, batch_size, src_int_pad, trg_int_pad)):

            targets_batch = process_decoder_input(targets_batch, trg_int_go, batch_size)

            # input [str_len, batch_size]
            src = torch.LongTensor(sources_batch).transpose(0, 1)
            trg = torch.LongTensor(targets_batch).transpose(0, 1)

            output = model(src, trg, 0)

            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim) #remove <GO>, shape:[out_len*batch_size, output_dim]
            trg = trg[1:].reshape(-1) #remove <GO>, shape: [trg len * batchsize]

            loss = criterion(output, trg)

            epoch_loss += loss.item()
            batch_num += 1
        return epoch_loss / batch_num

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

if __name__ == "__main__":

    batch_size = 32
    source_path = 'data/custom_source.txt'
    target_path = 'data/custom_target.txt'
    log_dir = './logs_dropout/'
    checkpoint_dir = './checkpoints_dropout/'
    resume = False
    start_epoch = 0

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

    # Split data to training and validation sets
    train_source = source_letter_ids[batch_size:]
    train_target = target_letter_ids[batch_size:]
    valid_source = source_letter_ids[:batch_size]
    valid_target = target_letter_ids[:batch_size]
    # (targets_batch, sources_batch, targets_lengths, sources_lengths) = next(
    #     get_batches(target_letter_ids, source_letter_ids, batch_size,
    #                 source_letter_to_int['<PAD>'],
    #                 target_letter_to_int['<PAD>']))

    # input_dim = len(source_letter_ids)
    # output_dim = len(target_letter_ids)
    input_dim = len(source_letter_to_int)
    output_dim = len(target_letter_to_int)
    enc_dim = 15
    dec_dim = 15
    hid_dim = 50
    n_layers = 2
    n_epochs = 300
    learning_rate = 0.001
    dropout= 0.2

    enc = Encoder(input_dim, enc_dim, hid_dim, n_layers, dropout)
    dec = Decoder(output_dim, dec_dim, hid_dim, n_layers, dropout)
    model = Seq2Seq(enc, dec, device).to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    if resume == True:
        resume_checkpoint = 'epoch_090_loss_0.196075.pt'
        model.load_state_dict(torch.load(checkpoint_dir+resume_checkpoint))
        start_epoch = 100

    # define writer
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir)

    # make checkpoint_dir
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    best_valid_loss = float('inf')
    best_epoch = 0
    for epoch in range(start_epoch, n_epochs):
        start_time = time.time()

        train_loss = train(model, optimizer, criterion, train_target , train_source, batch_size, source_letter_to_int['<PAD>'],
                        target_letter_to_int['<PAD>'], target_letter_to_int['<GO>'])
        valid_loss = evaluate(model, criterion, valid_target, valid_source, batch_size, source_letter_to_int['<PAD>'],
                        target_letter_to_int['<PAD>'], target_letter_to_int['<GO>'])

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # write summary
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/valid', valid_loss, epoch)

        if epoch % 10 == 0:
            torch.save(model.state_dict(), checkpoint_dir+"epoch_%03d_loss_%.06f.pt"%(epoch, valid_loss))
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_dir+'rnn-tut1-model.pt')

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
        print(f'\tBest Epoch: {best_epoch}')