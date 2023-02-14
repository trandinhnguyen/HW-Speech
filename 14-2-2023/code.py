import librosa
import numpy as np
import soundfile as sf

# param
sr = 16000
mono = True


def find_position(t, sr):
    return int(t*sr)


def concat_diphones(s1, s2, t1, t2, sr=16000, mono=True):
    sound1 = librosa.load(s1, sr=sr, mono=mono)[0]

    if s2 == None:
        return sound1

    sound2 = librosa.load(s2, sr=sr, mono=mono)[0]

    p1 = find_position(t1, sr)
    p2 = find_position(t2, sr)

    # cut sound
    sound1 = sound1[:p1]
    sound2 = sound2[p2:]

    # enery balancing
    e1 = np.sum(sound1 ** 2)
    e2 = np.sum(sound2 ** 2)

    if e1 >= e2:
        sound1 *= e2/e1
    else:
        sound2 *= e1/e2

    return np.concatenate((sound1, sound2))


def concat_sound(path, sr, *sound_list):
    sound = np.concatenate(sound_list)
    sf.write(path, sound, sr, subtype='PCM_16')


tran = concat_diphones(
    'wav/trân.wav',
    'wav/ần.wav',
    0.196,
    0.125,
    sr=sr,
    mono=mono
)

dinh = concat_diphones(
    'wav/đi.wav',
    'wav/ình.wav',
    0.225,
    0.126,
    sr=sr,
    mono=mono
)

nguyen = concat_diphones(
    'wav/nguy.wav',
    'wav/uyên.wav',
    0.243,
    0.144,
    sr=sr,
    mono=mono
)

concat_sound('wav/TrầnĐìnhNguyên.wav', sr, tran, dinh, nguyen)
