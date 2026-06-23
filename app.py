import streamlit as st
import requests

# Replace with your EC2 public IP
BLITZ_API = "http://34.207.213.111:8000"

st.title("🎙️ Blitz Transcription")
st.markdown("GPU-accelerated speech-to-text with intelligent model routing")

# File uploader
audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a", "flac"])

# Output format
output_format = st.selectbox("Output format", ["json", "text", "srt", "vtt"])

if audio_file and st.button("🚀 Transcribe"):
    with st.spinner("Transcribing..."):
        response = requests.post(
            f"{BLITZ_API}/transcribe",
            files={"file": (audio_file.name, audio_file.getvalue())},
            data={"format": output_format}
        )
    
    if response.status_code == 200:
        if output_format == "json":
            data = response.json()
            
            # Display results
            st.success(f"✅ Transcribed in {data['metrics']['total_processing_time_sec']:.2f}s")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Language", data['language']['code'].upper())
            col2.metric("Engine", data['engine'])
            col3.metric("RTF", f"{data['metrics']['real_time_factor']:.3f}")
            
            st.subheader("Transcription")
            st.write(data['text'])
            
            with st.expander("Full Response"):
                st.json(data)
        else:
            st.subheader("Output")
            st.code(response.text)
    else:
        st.error(f"Error: {response.status_code} - {response.text}")

# Health check
with st.sidebar:
    st.subheader("Server Status")
    try:
        health = requests.get(f"{BLITZ_API}/health", timeout=5).json()
        st.success("🟢 Connected")
        st.json(health)
    except:
        st.error("🔴 Cannot reach Blitz API")