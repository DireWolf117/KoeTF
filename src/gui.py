import datautils as utils 
import numpy as np
import models as mod
import pretty_midi
import subprocess
import libfmp.c1
import os

from matplotlib import pyplot as plt

from tkinter import *
from PIL import ImageTk, Image

global latent_vec
global encoder
global test_model
global start_measure
global threshold
global timesteps
global song_index
global interpolate_scale
timesteps = 64
threshold = 0.7
start_measure = []
latent_vec = []

#Create MIDI File base on the latent vector fed through
def ae_gen(value=None):
    global latent_vec
    global test_model
    global threshold
    latent_sample = np.empty((1, len(latent_vec)),float)
    for i in range(len(latent_vec)):
        latent_sample[0][i] = latent_vec[i].get()
    utils.MHE_to_txt(test_model(latent_sample), threshold.get())
    subprocess.call(['sh', 'src/gen_midi.sh']) 


def lstm_gen():
    global timesteps
    global test_model
    global start_measure
    global threshold
    start_measure_num = np.zeros((88), float)
    for i in range(len(start_measure)):
        if start_measure[i].get():
            start_measure_num[i] = 1.0
    x = mod.lstm_generate(test_model, start_measure_num, timesteps.get(), threshold.get())
    utils.MHE_to_txt(x, 0.2)
    subprocess.call(['sh', 'src/gen_midi.sh']) 

def get_seed():
    global encoder
    global latent_vec
    global song_index
    x = utils.data_from_npz_MHE("data/numpy_arrays/0_5632.npy")[song_index.get():song_index.get()+1].astype(np.float64)
    y = encoder(x)
    y = np.reshape(y, np.prod(y.shape))
    for i in range(len(y)):
        latent_vec[i].set(y[i])
    #x = utils.data_from_npz_MHE("data/numpy_arrays/0_5632.npy")
    #average = np.zeros((1, 16))
    #for i in range(np.size(x, axis=0) - 2):
    #    y = encoder(x[i:i+1])
    #    average = average + y
    #average = average / np.size(x, axis=0)
    #average = np.reshape(average, np.prod(average.shape))
    #print(average)
    #for i in range(len(average)):
    #    latent_vec[i].set(average[i])

def lstm_reset():
    for b in start_measure:
        b.set(False)

def represent_out_midi():
    represent_midi('out.mid')

def play_midi():
    subprocess.call(['sh', 'src/play_midi.sh'])

def gen_and_play():
    ae_gen()
    play_midi()

def interpolate(value=None):
    global latent_vec
    global interpolate_scale
    global song_index
    #really good song: 127
    x = utils.data_from_npz_MHE("data/numpy_arrays/0_5632.npy")[127:128].astype(np.float64)
    utils.MHE_to_txt(x, 0.7)
    xx = utils.data_from_npz_MHE("data/numpy_arrays/0_5632.npy")[song_index.get():song_index.get()+1].astype(np.float64)
    y = encoder(x)
    y = np.reshape(y, np.prod(y.shape))
    yy = encoder(xx)
    yy = np.reshape(yy, np.prod(yy.shape))
    for i in range(len(y)):
        temp = yy[i] + ((y[i] - yy[i]) * interpolate_scale.get())
        latent_vec[i].set(temp)
    
def gen_gui(in_model):
    
    global test_model
    global encoder
    global threshold
    is_generic = True

    root = Tk()
    root.title("Test GUI")

    #Check if model is LSTM or normal neural network
    for l in in_model.layers:
        if len(l.input.shape) == 3:
            is_generic = False

    if is_generic:
        root.geometry("900x400")
        global latent_vec
        global song_index
        global interpolate_scale
        test_model = mod.decoder(in_model)
        encoder = mod.encoder(in_model)
        for i in range(test_model.layers[0].input.shape[1]):
            v = DoubleVar()
            vertical = Scale(root, variable=v, from_=-1.0, to=1.0, resolution=0.001, orient=HORIZONTAL, length=200)
            vertical.grid(row= i // 4, column= i % 4)
            latent_vec.append(v)
        u = DoubleVar()
        vertical = Scale(root, variable = u, from_=0.0, to=1.0, resolution=0.01, orient=HORIZONTAL, width=16, length=200)
        threshold = u
        vertical.grid(row = 5, column = 0)
        g = IntVar()
        song_slide = Scale(root, variable = g, from_=0, to=127, resolution=1, orient=HORIZONTAL, width=16, length=200)
        song_index = g
        song_slide.grid(row = 7, column = 0)
        h = DoubleVar()
        interpol_slide = Scale(root, variable = h, from_=0.0, to=1.0, resolution=0.01, orient=HORIZONTAL, width=16, length=200, command = interpolate)
        interpolate_scale = h
        interpol_slide.grid(row = 8, column = 0)

        b=Button(root,text="Generate",command=ae_gen)
        b.grid(row = 5, column=1)
        c=Button(root,text="Show MIDI",command=represent_out_midi)
        c.grid(row = 6, column=0)
        d=Button(root,text="Play MIDI",command=play_midi)
        d.grid(row = 6, column=1)
        e=Button(root,text="Set Seed",command=get_seed)
        e.grid(row = 6, column=2)
        f=Button(root,text="Gen and Play",command=gen_and_play)
        f.grid(row = 6, column=3)
    else: 
        root.geometry("600x600")
        test_model = in_model
        switch_frame = Frame(root)
        switch_frame.grid(row = 0, column = 0)
        global start_measure
        global timesteps
        for i in range(88):
            v = BooleanVar()
            note_toggle = Radiobutton(switch_frame, text=str(i), variable = v, indicatoron=False, value=True, width=8)
            note_toggle.grid(row= i // 8, column= i % 8)
            start_measure.append(v)
        u = DoubleVar()
        e = IntVar()
        time_scale = Scale(root, variable = e, from_=0, to=512, resolution=16, orient=HORIZONTAL, width=16, length=500)
        timesteps = e
        time_scale.grid(row = 26, column = 0)
        threshold_scale = Scale(root, variable = u, from_=0.0, to=10.0, resolution=0.1, orient=HORIZONTAL, width=16, length=250)
        threshold = u
        threshold_scale.grid(row = 23, column = 0)
        b=Button(root,text="Generate",command=lstm_gen)
        b.grid(row = 22, column=0)
        c=Button(root,text="Show MIDI",command=represent_out_midi)
        c.grid(row = 24, column=0)
        d=Button(root,text="Play MIDI",command=play_midi)
        d.grid(row = 25, column=0)
        d=Button(root,text="Reset",command=lstm_reset)
        d.grid(row = 27, column=0)
    root.mainloop()


def midi_to_list(midi):
    """Convert a midi file to a list of note events

    Notebook: C1/C1S2_MIDI.ipynb

    Args:
        midi (str or pretty_midi.pretty_midi.PrettyMIDI): Either a path to a midi file or PrettyMIDI object

    Returns:
        score (list): A list of note events where each note is specified as
            ``[start, duration, pitch, velocity, label]``
    """

    if isinstance(midi, str):
        midi_data = pretty_midi.pretty_midi.PrettyMIDI(midi)
    elif isinstance(midi, pretty_midi.pretty_midi.PrettyMIDI):
        midi_data = midi
    else:
        raise RuntimeError('midi must be a path to a midi file or pretty_midi.PrettyMIDI')

    score = []

    for instrument in midi_data.instruments:
        for note in instrument.notes:
            start = note.start
            duration = note.end - start
            pitch = note.pitch
            velocity = note.velocity / 128.
            score.append([start, duration, pitch, velocity, instrument.name])
    return score

def represent_midi(path_to_midi):
    fn = os.path.join(path_to_midi)
    midi_data = pretty_midi.PrettyMIDI(fn)
    score = midi_to_list(midi_data)
    libfmp.c1.visualize_piano_roll(score, figsize=(20, 10), velocity_alpha=False)
    plt.show()

def show_validation_loss():
    loss_file = open("logs/test_accuracy.txt").readlines()
    loss = []
    for l in loss_file:
        loss.append(float(l))
    plt.plot(loss)
    plt.show()

