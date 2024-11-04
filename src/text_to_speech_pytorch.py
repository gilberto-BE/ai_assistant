# import torch
# import torchaudio

# torch.random.manual_seed(0)
# device = "cuda" if torch.cuda.is_available() else "cpu"


# # import IPython
# import matplotlib.pyplot as plt

# symbols = "_-!'(),.:;? abcdefghijklmnopqrstuvwxyz"
# look_up = {s: i for i, s in enumerate(symbols)}


# def text_to_speech(text):
#     bundle = torch.audio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH
#     processor = bundle.get_text_processor()
#     tacotron2 = bundle.get_tacotron2().to(device)

#     vocoder = bundle.get_vocoder().to(device)

#     with torch.inference_mode():
#         processed, lengths = processor(text)
#         processed = processed.to(device)
#         lengths = lengths.to(device)
#         spec, spec_lengths, _ = tacotron2.infer(processed, lengths)
#         waveforms, lengths = vocoder.infer(spec, spec_lengths)
#     return waveforms[0:1].cpu().detach()


# def text_to_sequence(text):
#     text = text.lower()
#     return [look_up[s] for s in text if s in symbols]


# text = "Hello world! Text to speech!"
# print(text_to_sequence(text))


# processor = torchaudio.pipelines.TACOTRON2_WAVERNN_CHAR_LJSPEECH.get_text_processor()
# text = "Hello world! Text to speech!"
# processed, lengths = processor(text)
# print(processed)
# print(lengths)

# bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH
# processor = bundle.get_text_processor()
# text = "Hello word! Text to spech!"
# with torch.inference_mode():
#     processed, lengths = processor(text)
# print(processed)
# print(lengths)


# bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH
# processor = bundle.get_text_processor()
# tacotron2 = bundle.get_tacotron2().to(device)
# text = "Hello world! Text to speech!"
# with torch.inference_mode():
#     processed, lengths = processor(text)
#     processed = processed.to(device)
#     lengths = lengths.to(device)
#     spec, _, _ = tacotron2.infer(processed, lengths)

# _ = plt.imshow(spec[0].cpu().detach(), aspect="auto", origin="lower")


# def plot():
#     fig, ax = plt.subplots(3, 1)
#     for i in range(3):
#         with torch.inference_mode():
#             spec, spec_lengths, _ = tacotron2.infer(processed, lengths)
#         print(spec[0].shape)
#         ax[i].imshow(spec[0].cpu().detach(), aspect="auto", origin="lower")


# plot()


import torch
import torchaudio
import matplotlib.pyplot as plt

torch.random.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

symbols = "_-!'(),.:;? abcdefghijklmnopqrstuvwxyz"
look_up = {s: i for i, s in enumerate(symbols)}

# Load models and processor once
bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH
processor = bundle.get_text_processor()
tacotron2 = bundle.get_tacotron2().to(device)
vocoder = bundle.get_vocoder().to(device)

torch.backends.cudnn.benchmark = True  # Enable cudnn benchmark for better performance


def text_to_speech(text):
    with torch.inference_mode():
        processed, lengths = processor(text)
        processed = processed.to(device)
        lengths = lengths.to(device)
        spec, spec_lengths, _ = tacotron2.infer(processed, lengths)
        waveforms, lengths = vocoder.infer(spec, spec_lengths)
    return waveforms[0:1].cpu().detach()


def plot():
    # Run inference once and reuse results
    with torch.inference_mode():
        spec, spec_lengths, _ = tacotron2.infer(processed, lengths)

    fig, ax = plt.subplots(3, 1)
    for i in range(3):
        ax[i].imshow(spec[0].cpu().detach(), aspect="auto", origin="lower")
        print(spec[0].shape)


# Example usage
text = "Hello world! Text to speech!"
processed, lengths = processor(text)
processed = processed.to(device)
lengths = lengths.to(device)

plot()
