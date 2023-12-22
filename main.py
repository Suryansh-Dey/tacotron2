import librosa.display
import torch
from model import Tacotron2
from hparams import create_hparams

char_vocab = {'<pad>': 0, '<unk>': 1, 'a': 2, 'b': 3, 'c': 4, 'd': 5, 'e': 6, 'f': 7, 'g': 8, 'h': 9,
              'i': 10, 'j': 11, 'k': 12, 'l': 13, 'm': 14, 'n': 15, 'o': 16, 'p': 17, 'q': 18, 'r': 19,
              's': 20, 't': 21, 'u': 22, 'v': 23, 'w': 24, 'x': 25, 'y': 26, 'z': 27, ' ': 28}

def text_to_indices(text, vocab):
    """
    Convert a sequence of characters to a sequence of indices based on a vocabulary.

    Args:
    - text (str): The input text sequence.
    - vocab (dict): A dictionary mapping characters to indices.

    Returns:
    - indices (list): A list of indices corresponding to the input text.
    """
    # Convert characters to lowercase if needed
    text = text.lower()

    # Replace unknown characters with a special token or handle them as needed
    indices = [vocab[char] if char in vocab else vocab['<unk>'] for char in text]

    return indices
state_dict_path = "C:/Users/BIT/OneDrive/Desktop/Projects/AIFriend/AIFriendBackend/resources/tacotron2_statedict.pt"
hparams=create_hparams()
model = Tacotron2(hparams)

# Load the pre-trained state dictionary into the model
state_dict = torch.load(state_dict_path, map_location=torch.device('cpu'))['state_dict']
model.load_state_dict(state_dict)
model.eval()
text = "I love you"
indices = text_to_indices(text,char_vocab)

# Use an embedding layer to convert indices to vectors
embedded_sequence = model.embedding(torch.tensor(indices))

# 3. Pass the embedded sequence through the encoder
input_lengths = torch.tensor([len(indices)])

# Pass the embedded sequence and input lengths through the encoder
embedded_sequence = embedded_sequence.unsqueeze(2)  # Add a channel dimension
embedded_sequence = embedded_sequence.permute(0, 2, 1)  # Swap dimensions to [1, 512, 10]

# Pass the adjusted input through the encoder
encoder_outputs = model.encoder(embedded_sequence, input_lengths)

# 4. Decode the encoder outputs to obtain mel spectrograms
mel_spectrogram = model.decoder.decode(encoder_outputs)

waveform = librosa.griffinlim(mel_spectrogram.numpy(), hop_length=256, win_length=800, n_fft=2048)

librosa.output.write_wav('output_audio.wav', waveform, 22050)

