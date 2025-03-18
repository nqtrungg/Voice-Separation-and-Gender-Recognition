import streamlit as st
import numpy as np
import librosa
import soundfile as sf
from io import BytesIO
from Utils import create_model
from sound_sep import SoundSeparation
import tempfile
import pyaudio
import wave

# Cấu hình Streamlit
st.set_page_config(page_title="Voice Separation & Gender", page_icon="🎙️", layout="centered")
st.title("🎙️ Voice Separation & Gender")
st.write("Ghi âm trực tiếp hoặc tải lên một file âm thanh (.wav) để tách giọng và nhận diện giới tính.")

# Khởi tạo mô hình
@st.cache_resource
def load_model():
    model = create_model()
    model.load_weights("results/model.h5")
    return model

if "sound_sep" not in st.session_state:
    st.session_state.sound_sep = SoundSeparation(config_path="config.yaml")
if "gender_model" not in st.session_state:
    st.session_state.gender_model = load_model()

# Hàm ghi âm
def record_audio(duration=10):
    """Ghi âm trực tiếp từ micro và trả về dữ liệu âm thanh."""
    CHUNK, FORMAT, CHANNELS, RATE = 1024, pyaudio.paInt16, 1, 16000
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    
    frames = [stream.read(CHUNK) for _ in range(0, int(RATE / CHUNK * duration))]
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
        wf = wave.open(temp_wav, "wb")
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))
        return temp_wav.name

# Hàm trích xuất đặc trưng
def extract_feature(file_name, **kwargs):
    """
    Trích xuất đặc trưng từ file âm thanh `file_name`.
    
    Các đặc trưng hỗ trợ:
        - MFCC (mfcc)
        - Chroma (chroma)
        - MEL Spectrogram (mel)
        - Contrast (contrast)
        - Tonnetz (tonnetz)
    
    Ví dụ:
    `features = extract_feature(path, mel=True, mfcc=True)`
    """
    try:
        mfcc = kwargs.get("mfcc")
        chroma = kwargs.get("chroma")
        mel = kwargs.get("mel")
        contrast = kwargs.get("contrast")
        tonnetz = kwargs.get("tonnetz")

        # Đọc file âm thanh
        X, sample_rate = librosa.load(file_name, sr=None)
        result = np.array([])

        # Tính toán STFT nếu cần cho chroma hoặc contrast
        stft = np.abs(librosa.stft(X)) if chroma or contrast else None

        # Trích xuất MFCC (n_mfcc=13 để phù hợp với model)
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))

        # Trích xuất Chroma
        if chroma:
            chroma_feature = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma_feature))

        # Trích xuất Mel Spectrogram
        if mel:
            mel_feature = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel_feature))

        # Trích xuất Spectral Contrast
        if contrast:
            contrast_feature = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, contrast_feature))

        # Trích xuất Tonnetz
        if tonnetz:
            tonnetz_feature = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
            result = np.hstack((result, tonnetz_feature))

        return result

    except Exception as e:
        st.error(f"❌ Lỗi khi trích xuất đặc trưng: {str(e)}")
        return None

# Chọn nguồn âm thanh
option = st.radio("📌 Chọn nguồn âm thanh:", ["🎤 Ghi âm trực tiếp", "📂 Tải file WAV"])

uploaded_file = None
if option == "🎤 Ghi âm trực tiếp" and st.button("🎙️ Bắt đầu ghi âm"):
    uploaded_file = record_audio()
    st.success(f"✅ Đã ghi âm xong!")
elif option == "📂 Tải file WAV":
    uploaded_file = st.file_uploader("📤 Chọn file WAV", type=["wav"])

# Xử lý file âm thanh
if uploaded_file:
    st.subheader("🔊 Audio Gốc")
    st.audio(uploaded_file, format="audio/wav")
    
    mixture, original_sample_rate = st.session_state.sound_sep.read_audio_file(uploaded_file)
    diarization, sources_hat = st.session_state.sound_sep.separate_sound(mixture, original_sample_rate)
    
    if sources_hat is None:
        st.error("⚠️ Không thể tách giọng. Thử lại với file khác.")
    else:
        st.subheader("📊 Kết quả")
        num_sources = sources_hat.data.shape[1]
        st.write(f"🔊 **Số giọng nói tách được:** {num_sources}")
        
        for i in range(num_sources):
            source = sources_hat[:, i]
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
                sf.write(temp_wav.name, source, original_sample_rate, format='wav')
                temp_path = temp_wav.name
            
            st.subheader(f"🗣️ Giọng {i+1}")
            st.audio(temp_path, format="audio/wav")
            
            features = extract_feature(temp_path, mel=True)
            
            if features is not None:
                features = features.reshape(1, -1)  # Định dạng đầu vào cho mô hình

                # Kiểm tra số lượng đặc trưng có khớp với model không
                expected_features = 128  # Model yêu cầu 128 đặc trưng
                if features.shape[1] != expected_features:
                    st.error(f"⚠️ Lỗi: Model yêu cầu {expected_features} đặc trưng, nhưng nhận được {features.shape[1]}.")
                else:
                    # Dự đoán giới tính
                    model = st.session_state.gender_model
                    male_prob = model.predict(features)[0][0]
                    female_prob = 1 - male_prob
                    gender = "🧑 Male" if male_prob > female_prob else "👩 Female"

                    # Hiển thị kết quả
                    st.write(f"**Giới tính dự đoán:** {gender}")
                    st.write(f"🔵 **Nam:** {male_prob * 100:.2f}%  |  🔴 **Nữ:** {female_prob * 100:.2f}%")
                
# streamlit run app.py