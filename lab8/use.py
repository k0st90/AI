from tensorflow import keras
import tensorflow as tf
import numpy as np

def CTCLoss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")  
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")  
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")  

    input_length = input_length * tf.ones((batch_len, 1), dtype=tf.int64)  
    label_length = label_length * tf.ones((batch_len, 1), dtype=tf.int64)  

    return keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)

model = keras.models.load_model("final_model.h5", custom_objects={"CTCLoss": CTCLoss})
print("✅ Model loaded successfully!")

frame_length, frame_step, fft_length = 256, 160, 384  

def preprocess_audio(file_path):
    file = tf.io.read_file(file_path)
    audio, _ = tf.audio.decode_wav(file)
    audio = tf.squeeze(audio, axis=-1)
    audio = tf.cast(audio, tf.float32)

    spectrogram = tf.signal.stft(audio, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length)
    spectrogram = tf.abs(spectrogram) ** 0.5

    means, stddevs = tf.math.reduce_mean(spectrogram, 1, keepdims=True), tf.math.reduce_std(spectrogram, 1, keepdims=True)
    spectrogram = (spectrogram - means) / (stddevs + 1e-10)

    spectrogram = tf.expand_dims(spectrogram, axis=0)  # Add batch dimension
    return spectrogram

characters = [x for x in "abcdefghijklmnopqrstuvwxyz'?! "]
char_to_num = keras.layers.StringLookup(vocabulary=characters, oov_token="")
num_to_char = keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)

def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    return [tf.strings.reduce_join(num_to_char(r)).numpy().decode("utf-8") for r in results]


def transcribe_audio(file_path):
    spectrogram = preprocess_audio(file_path)
    predictions = model.predict(spectrogram)
    decoded_text = decode_batch_predictions(predictions)[0]  
    return decoded_text

audio_file_path = "LJSpeech-1.1\wavs\LJ044-0217.wav"
transcription = transcribe_audio(audio_file_path)
print(f"📝 Transcription: {transcription}")

