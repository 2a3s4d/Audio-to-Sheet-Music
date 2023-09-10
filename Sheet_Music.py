import math as m
import numpy as np
import scipy.io.wavfile
from scipy.fftpack import fft
from random import randint
import matplotlib.pyplot as plt

HEADER = ['<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
          '<score-partwise version="4.0">',
          '<part-list>',
          '<score-part id="P1">',
          '<part-name>Music</part-name>',
          '</score-part>',
          '</part-list>',
          '<part id="P1">']

ATTRIBUTES = ['<attributes>',
              '<divisions>24</divisions>',
              '<time>',
              '<beats>4</beats>',
              '<beat-type>4</beat-type>',
              '</time>',
              '</attributes>']

FOOTER = ['</part>',
          '</score-partwise>']


NOTES = ["C0", "C#0", "D0", "D#0", "E0", "F0", "F#0", "G0",
         "G#0", "A0", "A#0", "B1", "C1", "C#1", "D1", "D#1", "E1", "F1", "F#1", "G1",
         "G#1", "A1", "A#1", "B1", "C2", "C#2", "D2", "D#2", "E2", "F2", "F#2", "G2",
         "G#2", "A2", "A#2", "B2", "C3", "C#3", "D3", "D#3", "E3", "F3", "F#3", "G3", 
         "G#3", "A3", "A#3", "B3", "C4", "C#4", "D4", "D#4", "E4", "F4", "F#4", "G4", 
         "G#4", "A4", "A#4", "B4", "C5", "C#5", "D5", "D#5", "E5", "F5", "F#5", "G5",
         "G#5", "A5", "A#5", "B5", "C6", "C#6", "D6", "D#6", "E6", "F6", "F#6", "G6",
         "G#6", "A6", "A#6", "B6", "C7", "C#7", "D7", "D#7", "E7", "F7", "F#7", "G7",
         "G#7", "A7", "A#7", "B7", "C8", "C#8", "D8", "D#8", "E8", "F8", "F#8", "G8",
         "G#8", "A8", "A#8", "B8"]

#NOTE_LENGTHS = ["eighth", "quarter", "half", "whole"]

NOTE_LENGTHS = {108 : ['whole', 'eighth'],
                96 : ['whole'],
                84 : ['half', 'quarter', 'eighth'],
                72 : ['half', 'quarter'],
                60 : ['half', 'eighth'],
                48 : ['half'],
                36 : ['quarter', 'eighth'],
                24 : ['quarter'],
                18 : ['eighth', '16th'],
                12 : ['eighth'],
                6 : ['16th']}

LENGTH_NOTES = {'whole' : 96,
                'half' : 48,
                'quarter': 24, 
                'eighth' : 12,
                '16th' : 6}


LOWEST_NOTE_FRQ = 16.35
MAX_NOTE_FRQ = 7902.13

INSTRUMENT_RANGES = {'Tenor Saxophone' : [103.83, 660],
                    'Piano' : [27.5, 4186.5]}

TIME_STEP = 0.2

def fix_out_of_tune(prevNote, curNote):
    return curNote > prevNote * 0.945 and curNote < prevNote * 1.055

def slow_down_basic (L):
    nL = []
    for e in L:
        nL.append(e)
        nL.append(e)
        
    return nL

def convert_one_chanel (rec):
    new_rec = []
    for frqs in rec:
        new_rec.append((frqs[0] + frqs[1]) / 2.0)
    
    return new_rec

def closest_pitch(frq: float):
    i = int(round(12 * m.log2(frq / LOWEST_NOTE_FRQ)))
    adj_pitch = LOWEST_NOTE_FRQ * pow(2, i / 12)
    return round(adj_pitch, 1)

def closest_note(frq: float):
    i = int(round(12 * m.log2(frq / LOWEST_NOTE_FRQ)))
    note = NOTES[i]
    return note

def lowest_note (notes):
    i = notes[0]
    
    for n in notes:
        i = min(i, n)

    return i
            

def cleanup (all_notes, inTune=True):
    cleaned_notes = []
    if (inTune):
        while (len(all_notes) > 0):
            
            if (all_notes[0][0] == 0):
                all_notes = all_notes[1:]
            else:
                cleaned_notes.append([closest_note(all_notes[0][0]), TIME_STEP])
                j = 1
                
                while (j < len(all_notes) and all_notes[0][0] == all_notes[j][0]):
                    cleaned_notes[-1][1] += TIME_STEP
                    j += 1
                all_notes = all_notes[j:]
            
        return cleaned_notes
    
    else:
        while (len(all_notes) > 0):
            
            if (all_notes[0][0] == 0):
                all_notes = all_notes[1:]
            else:
                cleaned_notes.append([closest_note(all_notes[0][0]), TIME_STEP])
                j = 1
                while (j < len(all_notes) and  fix_out_of_tune(all_notes[0][0], all_notes[j][0])):
                    cleaned_notes[-1][1] += TIME_STEP
                    j += 1 
            
                all_notes = all_notes[j:]
            
        return cleaned_notes
    
def cleanup_s (all_notes):
    cleaned_notes = []
    while (len(all_notes) > 0):
        
        if (all_notes[0][0] == 0):
            all_notes = all_notes[1:]
        else:
            cleaned_notes.append([all_notes[0][0], TIME_STEP])
            j = 1
            while (j < len(all_notes) and all_notes[0][0] == all_notes[j][0]):
                cleaned_notes[-1][1] += TIME_STEP
                j += 1 
        
            all_notes = all_notes[j:]
        
    return cleaned_notes

def write_to_xml (data, bpm, file_name=""):
    
    if (file_name == ""):
        file_name = str(randint(1000000, 9999999))
    sample_len = 0
    for note in data:
        sample_len += note[1]
    
    beats_used = 0
    q_note_div = 24
    beats_in_bar = 4 * q_note_div
    bps = bpm / 60
    num_beats = m.ceil(sample_len * bps)
    i = 1
    
    with open("Sheets/%s.xml" %(file_name), 'w') as f:
        for line in HEADER:
            f.write(line + "\n")
            
        f.write('<measure number="1">\n')

        for line in ATTRIBUTES:
            f.write(line + "\n")
                    
        end_bar = False
        note_len = 0
        note_len_hold = 0
        for note in data:
            note_len = int(max(0.5, int(bps * note[1])) * q_note_div)
            if (beats_used == beats_in_bar):
                i += 1
                f.write('</measure>\n')
                f.write('<measure number="%s">\n' %(i))
                beats_used = 0
        
            if (beats_used + note_len <= beats_in_bar):
                beats_used += note_len

                f.write('<note>\n<pitch>\n')
                f.write('<step>%s</step>\n' %(note[0][0]))
                if(len(note[0]) == 3):
                    f.write('<alter>1</alter>\n')
                f.write('<octave>%s</octave>\n' %(note[0][-1]))
                f.write('</pitch>\n')
                f.write('<duration>%s</duration>\n' %(note_len))
                f.write('<type>%s</type>\n' %(NOTE_LENGTHS[note_len][0]))
                if (len(NOTE_LENGTHS[note_len]) != 1):
                    f.write("<dot></dot>\n")
                
                f.write('</note>\n')
                
            else: # 'overhang'
                note_len_hold = beats_used + note_len - beats_in_bar
                
                note_len -= note_len_hold
                
                print (beats_used + note_len, note_len_hold)
                
                if (len(NOTE_LENGTHS[note_len]) == 1):
                    f.write('<note>\n<pitch>\n')
                    f.write('<step>%s</step>\n' %(note[0][0]))
                    if(len(note[0]) == 3):
                        f.write('<alter>1</alter>\n')
                    f.write('<octave>%s</octave>\n' %(note[0][-1]))
                    f.write('</pitch>\n')
                    f.write('<duration>%s</duration>\n' %(note_len))
                    f.write('<type>%s</type>\n' %(NOTE_LENGTHS[note_len][0]))
                    f.write('<notations>\n')
                    f.write('<tied type="start"/>\n')
                    f.write('</notations>\n')
                    f.write('</note>\n')

                else:
                    j = 0
                    for type in NOTE_LENGTHS[note_len]:  
                        f.write('<note>\n<pitch>\n')
                        f.write('<step>%s</step>\n' %(note[0][0]))
                        if(len(note[0]) == 3):
                            f.write('<alter>1</alter>\n')
                        f.write('<octave>%s</octave>\n' %(note[0][-1]))
                        f.write('</pitch>\n')
                        f.write('<duration>%s</duration>\n' %(LENGTH_NOTES[type]))
                        f.write('<type>%s</type>\n' %(type))
                        
                        if (j == 0):
                            f.write('<notations>\n')
                            f.write('<tied type="start"/>\n')
                            f.write('</notations>\n')
                            j += 1
                        
                        else:
                            f.write('<notations>\n')
                            f.write('<tied type="stop"/>\n')
                            f.write('<tied type="start"/>\n')
                            f.write('</notations>\n')
                        note_len -= LENGTH_NOTES[type]
                        f.write('</note>\n')
                
                f.write('</measure>\n')
                i += 1
                f.write('<measure number="%s">' %(i))
                note_len = note_len_hold
                beats_used = note_len
                
                f.write('<note>\n<pitch>\n')
                f.write('<step>%s</step>\n' %(note[0][0]))
                if(len(note[0]) == 3):
                    f.write('<alter>1</alter>\n')
                f.write('<octave>%s</octave>\n' %(note[0][-1]))
                f.write('</pitch>\n')
                f.write('<duration>%s</duration>\n' %(note_len))
                
                f.write('<type>%s</type>\n' %(NOTE_LENGTHS[note_len][0]))
                if (len(NOTE_LENGTHS[note_len]) != 1):
                    f.write('<dot></dot>\n')
                f.write('</note>\n')


        if (beats_used < 96):
            note_len = beats_in_bar - beats_used
            if (len(NOTE_LENGTHS[note_len]) == 1):
                f.write('<note>\n')
                f.write('<rest></rest>\n')
                f.write('<duration>%s</duration>\n' %(note_len))
                f.write('<type>%s</type>\n' %(NOTE_LENGTHS[note_len][0]))
                f.write('</note>\n')

            else:
                j = 0
                for type in NOTE_LENGTHS[note_len]:  
                    f.write('<note>\n')
                    f.write('<rest></rest>\n')
                    f.write('<duration>%s</duration>\n' %(LENGTH_NOTES[type]))
                    f.write('<type>%s</type>\n' %(type))
                    f.write('</note>\n')

        f.write('</measure>\n')
        for line in FOOTER:
            f.write(line + "\n")
    return num_beats

def tot_time (all_notes):
    t = 0
    for note in all_notes:
        t += note[1]
        
    return round(t, 2)

def audio_notes(recording, instrument):
    sampleFreq, full_recording = scipy.io.wavfile.read(recording)
    
    sampleDur = len(full_recording) / sampleFreq
    
    myRecording = []
    final_f = 0
    av_max = 0
    n = 0
    print(type(full_recording[0]))
    if (type(full_recording[0]) != np.int16):
        full_recording = convert_one_chanel(full_recording)[:len(full_recording) // 2]
    
    
    #full_recording = slow_down_basic(full_recording)
    
    tot_len = len(full_recording)
    for i in range(int(sampleDur / TIME_STEP)):
        myRecording.append([])
        
        for j in range(int(tot_len / (sampleDur / TIME_STEP))):
            myRecording[i].append(full_recording[j])
            final_f = j

        full_recording = full_recording[final_f:]
    
    notes = []
    j = 0
    for frqs in myRecording:
        
        notes.append([0, 0])
        timeX = np.arange(0, sampleFreq / 2, sampleFreq / len(frqs))
        absFreqSpectrum = abs(fft(frqs))

        minFrq = int(INSTRUMENT_RANGES[instrument][0] / timeX[1])
        maxFrq = int(INSTRUMENT_RANGES[instrument][1] / timeX[1])
        timeX = timeX[minFrq : maxFrq]
        absFreqSpectrum = absFreqSpectrum[minFrq : maxFrq]
        av_max += (max(absFreqSpectrum))
        n += 1
        pos_notes = []
        pos_notes_sec = []
        for i in range(len(timeX)):
            if (INSTRUMENT_RANGES[instrument][0] < closest_pitch(timeX[i]) and INSTRUMENT_RANGES[instrument][1] > closest_pitch(timeX[i])):
                
                if (absFreqSpectrum[i] > 2 * 10 ** 6 and closest_pitch(timeX[i]) not in pos_notes and
                    (timeX[i] > timeX[max(i - 1, 0)] and timeX[i] < timeX[min(i + 1, len(timeX) - 1)])):
                        #peaks  
                    pos_notes.append(closest_pitch(timeX[i]))
            
                    
                #elif (absFreqSpectrum[i] > 10 ** 6 and closest_pitch(timeX[i]) not in pos_notes_sec):
                #    pos_notes_sec.append(closest_pitch(timeX[i]))

                    
        for n in list(pos_notes): # note
            for sn in pos_notes_sec: # secondary note
                if (n > sn and closest_pitch(n) % closest_pitch(sn) < 0.1):
                    pos_notes.append(sn)
                    
        if (len(pos_notes) != 0):
            notes[j][0] = lowest_note(pos_notes)
    
        else:
            notes[j][0] = 0
        notes[j][1] = TIME_STEP
        '''
        if len(timeX != 0):
            print(pos_notes)
            print(pos_notes_sec)
            plt.plot(timeX, absFreqSpectrum) #len(myRecording)//2
            plt.ylabel('|X(n)|')
            plt.xlabel('frequency[Hz]')
            plt.show()
        '''
        j += 1
    print(av_max / n)
    return notes

def whole_hog (filepath: str, instrument: str, bpm: int, filename: str, inTune=True):
    raw_data = audio_notes(filepath, instrument)
    cleaned_data = cleanup(raw_data, inTune=inTune)
    write_to_xml(cleaned_data, bpm, filename)



if (__name__ == '__main__'):
    whole_hog("*****.wav", "INSTRUMENT NAME", 80, "FILENAME")
