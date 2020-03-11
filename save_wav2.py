import wave

import pyaudio

chunk = 4096  # Record in chunks of 1024 samples
sample_format = pyaudio.paInt16  # 16 bits per sample
channels = 1
fs = 16000  # Record at 44100 samples per second

seconds1 = 10

filename1 = "D:/level4_sem1/project_interim2/wavFiles/output1.wav"

p = pyaudio.PyAudio()  # Create an interface to PortAudio

# playsound.playsound('D:/level4_sem1/project_interim2/mp3/start_voice.mp3', True)

print('Recording')

stream = p.open(format=sample_format,
                channels=channels,
                rate=fs,
                frames_per_buffer=chunk,
                input=True)

frames1 = []  # Initialize array to store frames


# Store data in chunks for 3 seconds
for i in range(0, int(fs / chunk * seconds1)):
    data1 = stream.read(chunk)
    frames1.append(data1)

# Stop and close the stream
stream.stop_stream()
stream.close()
# Terminate the PortAudio interface
p.terminate()

print('Finished recording')

# Save the recorded data as a WAV file
wf = wave.open(filename1, 'wb')
wf.setnchannels(channels)
wf.setsampwidth(p.get_sample_size(sample_format))
wf.setframerate(fs)
wf.writeframes(b''.join(frames1))
wf.close()
