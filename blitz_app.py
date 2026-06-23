import streamlit as st
import requests
import time
import threading
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================
BLITZ_API = "http://34.207.213.111:8000"

st.set_page_config(page_title="Blitz ASR", page_icon="🎙️", layout="wide")

# =============================================================================
# Custom CSS
# =============================================================================
st.markdown("""
<style>
    .stMetric { background: #f8f9fa; padding: 10px; border-radius: 8px; }
    .success-box { background: #d4edda; padding: 15px; border-radius: 8px; border-left: 4px solid #28a745; }
    .benchmark-box { background: #d1ecf1; padding: 15px; border-radius: 8px; border-left: 4px solid #17a2b8; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Sidebar — Server Status & Settings
# =============================================================================
with st.sidebar:
    st.subheader("🖥️ Server Status")
    api_url = st.text_input("API URL", value=BLITZ_API)
    BLITZ_API = api_url
    
    try:
        health = requests.get(f"{BLITZ_API}/health", timeout=5).json()
        st.success("🟢 Connected")
        with st.expander("Server Details"):
            st.json(health)
    except Exception as e:
        st.error("🔴 Cannot reach Blitz API")
        st.caption(f"Error: {str(e)}")

# =============================================================================
# Main Tabs
# =============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎙️ Transcribe", "🗣️ Speaker Diarization", "🌍 Multi-Language Testing", "📊 Performance Benchmark", "🖥️ System Monitor"])

# =============================================================================
# TAB 1: Single File Transcription
# =============================================================================
with tab1:
    st.header("🎙️ Transcribe Audio")
    
    col_upload, col_options = st.columns([2, 1])
    
    with col_upload:
        audio_file = st.file_uploader(
            "Upload an audio file",
            type=["mp3", "wav", "m4a", "flac"],
            help="Supported formats: MP3, WAV, M4A, FLAC",
            key="single_upload"
        )
        if audio_file:
            st.audio(audio_file, format=f"audio/{audio_file.name.split('.')[-1]}")
    
    with col_options:
        output_format = st.selectbox("Output format", ["json", "text", "srt", "vtt"])
        language_hint = st.text_input("Language hint (optional)", placeholder="e.g., en, fr, de")
    
    if audio_file and st.button("🚀 Transcribe", type="primary", use_container_width=True, key="btn_transcribe"):
        
        file_size_mb = len(audio_file.getvalue()) / (1024 * 1024)
        estimated_time = max(2.0, file_size_mb * 1.5)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        stages = [
            ("📤 Uploading audio file...", 0.05),
            ("🔍 Detecting language (Whisper)...", 0.25),
            ("🎯 Routing to optimal model...", 0.35),
            ("🔇 Voice activity detection...", 0.40),
            ("⚡ GPU inference running...", 0.55),
            ("📝 Generating transcription...", 0.80),
            ("📊 Computing metrics...", 0.90),
        ]
        
        result = {"response": None, "error": None, "done": False}
        
        def call_api():
            try:
                files = {"file": (audio_file.name, audio_file.getvalue())}
                data = {"format": output_format}
                if language_hint:
                    data["language_hint"] = language_hint
                response = requests.post(f"{BLITZ_API}/transcribe", files=files, data=data, timeout=300)
                result["response"] = response
            except Exception as e:
                result["error"] = str(e)
            finally:
                result["done"] = True
        
        thread = threading.Thread(target=call_api)
        thread.start()
        
        stage_idx = 0
        start_time = time.time()
        
        while not result["done"]:
            elapsed = time.time() - start_time
            progress = min(0.95, elapsed / estimated_time)
            while stage_idx < len(stages) - 1 and progress > stages[stage_idx + 1][1]:
                stage_idx += 1
            status_text.markdown(f"**{stages[stage_idx][0]}**")
            progress_bar.progress(progress)
            time.sleep(0.1)
        
        progress_bar.progress(1.0)
        status_text.empty()
        thread.join()
        
        if result["error"]:
            st.error(f"❌ Error: {result['error']}")
        elif result["response"] and result["response"].status_code == 200:
            if output_format == "json":
                data = result["response"].json()
                
                st.markdown(f"""
                <div class="success-box">
                    ✅ Transcribed <b>{data['duration_sec']:.1f}s</b> of audio in 
                    <b>{data['metrics']['total_processing_time_sec']:.2f}s</b> 
                    (RTF: {data['metrics']['real_time_factor']:.3f})
                </div>
                """, unsafe_allow_html=True)
                st.write("")
                
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Language", data['language']['code'].upper())
                col2.metric("Confidence", f"{data['language']['confidence']:.1%}")
                col3.metric("Engine", data['engine'].replace("tier", "Tier "))
                col4.metric("Duration", f"{data['duration_sec']:.1f}s")
                col5.metric("Cost/1K min", f"${data['metrics']['cost_per_1000_min_audio_usd']['spot']:.2f}")
                
                st.subheader("📝 Transcription")
                st.write(data['text'])
                
                with st.expander("🔍 Language Detection Details"):
                    lang_data = data['language']
                    st.write(f"**Method:** {lang_data['detection_method']} ({lang_data['detection_model']})")
                    st.write("**Top languages:**")
                    for code, conf in lang_data['top_languages'].items():
                        st.write(f"  - {code}: {conf:.6f}")
                
                with st.expander("⏱️ Timestamps"):
                    timestamps = data.get('timestamps')
                    if timestamps and isinstance(timestamps, dict):
                        word_ts = timestamps.get('word')
                        if word_ts and len(word_ts) > 0:
                            for word_info in word_ts:
                                start = word_info.get('start', 0)
                                end = word_info.get('end', 0)
                                word = word_info.get('word', '')
                                st.write(f"`{start:.2f}s - {end:.2f}s` → **{word}**")
                        else:
                            st.info("No word-level timestamps in this response")
                    else:
                        st.info("Timestamps not available (may vary by engine/tier)")
                
                with st.expander("📊 Performance Metrics"):
                    metrics = data['metrics']
                    mcol1, mcol2 = st.columns(2)
                    with mcol1:
                        st.write(f"**Total processing:** {metrics['total_processing_time_sec']:.3f}s")
                        st.write(f"**Language detection:** {metrics['language_detection_time_ms']:.1f}ms")
                        st.write(f"**Transcription:** {metrics['transcription_time_sec']:.3f}s")
                    with mcol2:
                        st.write(f"**Real-time factor:** {metrics['real_time_factor']:.4f}")
                        st.write(f"**Instance:** {metrics['instance_type']}")
                        st.write(f"**Chunk approach:** {metrics['chunk_approach']}")
                
                with st.expander("🔎 Full JSON Response"):
                    st.json(data)
            else:
                st.subheader(f"📝 Output ({output_format.upper()})")
                st.code(result["response"].text, language=None)
                st.download_button(
                    f"⬇️ Download .{output_format}",
                    data=result["response"].text,
                    file_name=f"{audio_file.name.rsplit('.', 1)[0]}.{output_format}",
                    mime="text/plain"
                )
        else:
            st.error(f"❌ Server returned {result['response'].status_code}: {result['response'].text}")

# =============================================================================
# TAB 2: Speaker Diarization
# =============================================================================
with tab2:
    st.header("🗣️ Speaker Diarization")
    st.markdown("Upload a multi-speaker audio file to identify **who said what**. Uses NeMo TitaNet for speaker embeddings + clustering.")

    dcol1, dcol2 = st.columns([2, 1])

    with dcol1:
        diar_file = st.file_uploader(
            "Upload audio with multiple speakers",
            type=["mp3", "wav", "m4a", "flac"],
            help="Best results with 2+ speakers and clear audio",
            key="diar_upload"
        )
        if diar_file:
            st.audio(diar_file, format=f"audio/{diar_file.name.split('.')[-1]}")

    with dcol2:
        diar_num_speakers = st.number_input(
            "Number of speakers (0 = auto-detect)",
            min_value=0, max_value=10, value=0,
            help="Set to 0 for auto-detection, or specify exact number for better accuracy"
        )
        diar_format = st.selectbox("Output format", ["json", "text"], key="diar_format")

    if diar_file and st.button("🗣️ Diarize & Transcribe", type="primary", use_container_width=True, key="btn_diarize"):

        progress_bar = st.progress(0)
        status_text = st.empty()

        stages = [
            ("📤 Uploading audio...", 0.05),
            ("🔍 Detecting language...", 0.20),
            ("⚡ Transcribing with timestamps...", 0.40),
            ("🗣️ Extracting speaker embeddings (TitaNet)...", 0.60),
            ("🔗 Clustering speakers...", 0.75),
            ("📝 Merging speakers with transcript...", 0.90),
        ]

        result = {"response": None, "error": None, "done": False}

        def call_diarize_api():
            try:
                files = {"file": (diar_file.name, diar_file.getvalue())}
                data = {"format": diar_format}
                if diar_num_speakers > 0:
                    data["num_speakers"] = str(diar_num_speakers)
                result["response"] = requests.post(
                    f"{BLITZ_API}/diarize", files=files, data=data, timeout=300
                )
            except Exception as e:
                result["error"] = str(e)
            finally:
                result["done"] = True

        thread = threading.Thread(target=call_diarize_api)
        thread.start()

        stage_idx = 0
        start_time = time.time()
        file_size_mb = len(diar_file.getvalue()) / (1024 * 1024)
        estimated_time = max(3.0, file_size_mb * 2.0)

        while not result["done"]:
            elapsed = time.time() - start_time
            progress = min(0.95, elapsed / estimated_time)
            while stage_idx < len(stages) - 1 and progress > stages[stage_idx + 1][1]:
                stage_idx += 1
            status_text.markdown(f"**{stages[stage_idx][0]}**")
            progress_bar.progress(progress)
            time.sleep(0.1)

        progress_bar.progress(1.0)
        status_text.empty()
        thread.join()

        if result["error"]:
            st.error(f"❌ Error: {result['error']}")
        elif result["response"] and result["response"].status_code == 200:
            data = result["response"].json()

            if diar_format == "text":
                st.subheader("📝 Speaker-Labeled Transcript")
                st.code(data.get("transcript", ""), language=None)
            else:
                # JSON format
                diar_data = data.get("diarization", {})

                st.success(f"✅ Detected **{diar_data.get('num_speakers', '?')} speakers** in {data.get('duration_sec', 0):.1f}s of audio")

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Speakers", diar_data.get("num_speakers", "?"))
                col2.metric("Engine", data.get("engine", "?"))
                col3.metric("Diarization Time", f"{diar_data.get('processing_time_sec', 0):.2f}s")
                col4.metric("Total Time", f"{data.get('metrics', {}).get('total_processing_time_sec', 0):.2f}s")

                # Speaker-labeled transcript
                st.subheader("📝 Speaker Transcript")
                transcript_text = diar_data.get("transcript", "")
                if transcript_text:
                    for line in transcript_text.split("\n"):
                        if "speaker_0" in line:
                            st.markdown(f"🔵 {line}")
                        elif "speaker_1" in line:
                            st.markdown(f"🟠 {line}")
                        elif "speaker_2" in line:
                            st.markdown(f"🟢 {line}")
                        else:
                            st.markdown(f"⚪ {line}")

                with st.expander("📊 Segments Detail"):
                    segments = diar_data.get("segments", [])
                    st.dataframe(segments, use_container_width=True, hide_index=True)

                with st.expander("🔎 Full JSON Response"):
                    st.json(data)
        else:
            st.error(f"❌ Server returned {result['response'].status_code}: {result['response'].text}")

# =============================================================================
# TAB 3: Multi-Language Testing
# =============================================================================
with tab3:
    st.header("🌍 Multi-Language Testing")
    st.markdown("Upload multiple audio files in different languages to verify model routing and language detection accuracy.")
    
    # Predefined language list (matches notebook's test set)
    available_languages = [
        "english", "french", "german", "spanish", "italian",
        "portuguese", "dutch", "polish", "czech", "russian",
        "swedish", "danish", "finnish", "hungarian", "romanian",
        "greek", "bulgarian", "croatian", "slovak", "slovenian",
        "ukrainian", "estonian", "latvian", "lithuanian",
        "japanese", "chinese", "korean", "arabic", "hindi", "turkish"
    ]
    
    # Expected tier routing
    tier1_languages = {"english"}  # en → Tier 1
    tier2_languages = {"french", "german", "spanish", "italian", "portuguese", "dutch", 
                       "polish", "czech", "russian", "swedish", "danish", "finnish",
                       "hungarian", "romanian", "greek", "bulgarian", "croatian",
                       "slovak", "slovenian", "ukrainian", "estonian", "latvian", "lithuanian"}
    # Everything else → Tier 3
    
    multi_files = st.file_uploader(
        "Upload audio files (name them by language, e.g., french.mp3, german.wav)",
        type=["mp3", "wav", "m4a", "flac"],
        accept_multiple_files=True,
        key="multi_upload"
    )
    
    if multi_files:
        st.write(f"**{len(multi_files)} files uploaded:**")
        file_list = ", ".join([f.name for f in multi_files])
        st.caption(file_list)
    
    if multi_files and st.button("🌍 Run Multi-Language Test", type="primary", use_container_width=True, key="btn_multi"):
        
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, audio in enumerate(multi_files):
            progress = (idx) / len(multi_files)
            progress_bar.progress(progress)
            status_text.markdown(f"**Processing {idx+1}/{len(multi_files)}: {audio.name}...**")
            
            start = time.time()
            try:
                response = requests.post(
                    f"{BLITZ_API}/transcribe",
                    files={"file": (audio.name, audio.getvalue())},
                    data={"format": "json"},
                    timeout=120
                )
                elapsed = time.time() - start
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Determine expected tier from filename
                    file_lang = audio.name.rsplit('.', 1)[0].lower()
                    if file_lang in tier1_languages:
                        expected_tier = "tier1"
                    elif file_lang in tier2_languages:
                        expected_tier = "tier2"
                    else:
                        expected_tier = "tier3"
                    
                    actual_tier = data.get('engine', 'unknown')
                    routing_correct = (actual_tier == expected_tier)
                    
                    results.append({
                        "File": audio.name,
                        "Detected Language": data['language']['code'].upper(),
                        "Confidence": f"{data['language']['confidence']:.1%}",
                        "Engine": actual_tier,
                        "Expected Tier": expected_tier,
                        "Routing ✓": "✅" if routing_correct else "❌",
                        "Duration (s)": f"{data['duration_sec']:.1f}",
                        "Processing (s)": f"{elapsed:.2f}",
                        "RTF": f"{data['metrics']['real_time_factor']:.4f}",
                        "Text Preview": data['text'][:80] + "..." if len(data['text']) > 80 else data['text']
                    })
                else:
                    results.append({
                        "File": audio.name,
                        "Detected Language": "ERROR",
                        "Confidence": "-",
                        "Engine": "-",
                        "Expected Tier": "-",
                        "Routing ✓": "❌",
                        "Duration (s)": "-",
                        "Processing (s)": f"{elapsed:.2f}",
                        "RTF": "-",
                        "Text Preview": f"HTTP {response.status_code}"
                    })
            except Exception as e:
                results.append({
                    "File": audio.name,
                    "Detected Language": "ERROR",
                    "Confidence": "-",
                    "Engine": "-",
                    "Expected Tier": "-",
                    "Routing ✓": "❌",
                    "Duration (s)": "-",
                    "Processing (s)": "-",
                    "RTF": "-",
                    "Text Preview": str(e)[:80]
                })
        
        progress_bar.progress(1.0)
        status_text.empty()
        
        # Display results table
        st.subheader("📋 Results")
        st.dataframe(results, use_container_width=True, hide_index=True)
        
        # Summary metrics
        successful = [r for r in results if r["Detected Language"] != "ERROR"]
        routing_correct_count = sum(1 for r in results if r["Routing ✓"] == "✅")
        
        st.subheader("📊 Summary")
        scol1, scol2, scol3, scol4 = st.columns(4)
        scol1.metric("Files Tested", len(results))
        scol2.metric("Successful", len(successful))
        scol3.metric("Routing Accuracy", f"{routing_correct_count}/{len(results)}")
        scol4.metric("Languages Detected", len(set(r["Detected Language"] for r in successful)))
        
        # Tier distribution
        st.subheader("🎯 Tier Distribution")
        tier_counts = {}
        for r in successful:
            tier = r["Engine"]
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
        
        for tier, count in sorted(tier_counts.items()):
            st.write(f"**{tier}**: {count} files")
        
        # Download results as CSV
        import io
        import csv
        
        csv_buffer = io.StringIO()
        writer = csv.DictWriter(csv_buffer, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
        
        st.download_button(
            "⬇️ Download Results CSV",
            data=csv_buffer.getvalue(),
            file_name="multi_language_results.csv",
            mime="text/csv"
        )

# =============================================================================
# TAB 4: Performance Benchmark
# =============================================================================
with tab4:
    st.header("📊 Performance Benchmark")
    st.markdown("Upload multiple audio files to measure throughput, latency, and cost efficiency.")
    
    bench_files = st.file_uploader(
        "Upload audio files for benchmarking",
        type=["mp3", "wav", "m4a", "flac"],
        accept_multiple_files=True,
        key="bench_upload"
    )
    
    # Benchmark settings
    bcol1, bcol2, bcol3 = st.columns(3)
    with bcol1:
        concurrency = st.selectbox("Concurrency mode", ["Sequential", "Parallel (2)", "Parallel (4)", "Parallel (8)"])
    with bcol2:
        num_iterations = st.number_input("Iterations per file", min_value=1, max_value=10, value=1)
    with bcol3:
        warmup = st.checkbox("Warmup run (discard first result)", value=True)
    
    if bench_files:
        st.write(f"**{len(bench_files)} files loaded** | Total: {sum(len(f.getvalue()) for f in bench_files) / (1024*1024):.1f} MB")
    
    if bench_files and st.button("⚡ Run Benchmark", type="primary", use_container_width=True, key="btn_bench"):
        
        concurrency_level = {"Sequential": 1, "Parallel (2)": 2, "Parallel (4)": 4, "Parallel (8)": 8}[concurrency]
        
        benchmark_results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Build list of all requests to make
        all_requests = []
        for iteration in range(num_iterations + (1 if warmup else 0)):
            for audio in bench_files:
                all_requests.append({
                    "file": audio,
                    "iteration": iteration,
                    "is_warmup": warmup and iteration == 0
                })
        
        total_requests = len(all_requests)
        completed = [0]  # mutable counter for thread safety
        
        def transcribe_file(request_info):
            audio = request_info["file"]
            start = time.time()
            try:
                response = requests.post(
                    f"{BLITZ_API}/transcribe",
                    files={"file": (audio.name, audio.getvalue())},
                    data={"format": "json"},
                    timeout=300
                )
                elapsed = time.time() - start
                
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "file": audio.name,
                        "iteration": request_info["iteration"],
                        "is_warmup": request_info["is_warmup"],
                        "language": data['language']['code'].upper(),
                        "engine": data.get('engine', 'unknown'),
                        "duration_sec": data['duration_sec'],
                        "processing_sec": elapsed,
                        "rtf": data['metrics']['real_time_factor'],
                        "rtfx": 1 / data['metrics']['real_time_factor'] if data['metrics']['real_time_factor'] > 0 else 0,
                        "lang_detection_ms": data['metrics']['language_detection_time_ms'],
                        "transcription_sec": data['metrics']['transcription_time_sec'],
                        "cost_spot": data['metrics']['cost_per_1000_min_audio_usd']['spot'],
                        "success": True
                    }
                else:
                    return {
                        "file": audio.name, "iteration": request_info["iteration"],
                        "is_warmup": request_info["is_warmup"],
                        "success": False, "error": f"HTTP {response.status_code}"
                    }
            except Exception as e:
                return {
                    "file": audio.name, "iteration": request_info["iteration"],
                    "is_warmup": request_info["is_warmup"],
                    "success": False, "error": str(e)
                }
        
        # Run benchmark
        total_start = time.time()
        
        if concurrency_level == 1:
            # Sequential
            for idx, req in enumerate(all_requests):
                progress_bar.progress((idx) / total_requests)
                label = " (warmup)" if req["is_warmup"] else ""
                status_text.markdown(f"**Processing {idx+1}/{total_requests}: {req['file'].name}{label}...**")
                result = transcribe_file(req)
                benchmark_results.append(result)
        else:
            # Parallel
            status_text.markdown(f"**Running {total_requests} requests with concurrency={concurrency_level}...**")
            with ThreadPoolExecutor(max_workers=concurrency_level) as executor:
                futures = {executor.submit(transcribe_file, req): req for req in all_requests}
                for idx, future in enumerate(as_completed(futures)):
                    progress_bar.progress((idx + 1) / total_requests)
                    result = future.result()
                    benchmark_results.append(result)
        
        total_wall_time = time.time() - total_start
        progress_bar.progress(1.0)
        status_text.empty()
        
        # Filter out warmup results
        actual_results = [r for r in benchmark_results if not r.get("is_warmup", False) and r.get("success", False)]
        warmup_results = [r for r in benchmark_results if r.get("is_warmup", False)]
        failed_results = [r for r in benchmark_results if not r.get("success", False) and not r.get("is_warmup", False)]
        
        if not actual_results:
            st.error("No successful results to report.")
        else:
            # Summary banner
            total_audio_sec = sum(r["duration_sec"] for r in actual_results)
            total_proc_sec = sum(r["processing_sec"] for r in actual_results)
            avg_rtf = sum(r["rtf"] for r in actual_results) / len(actual_results)
            avg_rtfx = sum(r["rtfx"] for r in actual_results) / len(actual_results)
            throughput = total_audio_sec / total_wall_time
            
            st.markdown(f"""
            <div class="benchmark-box">
                📊 <b>Benchmark Complete</b> — {len(actual_results)} files processed in {total_wall_time:.1f}s wall time
                {"<br>⚠️ " + str(len(warmup_results)) + " warmup runs discarded" if warmup_results else ""}
                {"<br>❌ " + str(len(failed_results)) + " failed" if failed_results else ""}
            </div>
            """, unsafe_allow_html=True)
            st.write("")
            
            # Key metrics
            st.subheader("🏆 Key Metrics")
            kcol1, kcol2, kcol3, kcol4, kcol5 = st.columns(5)
            kcol1.metric("Total Audio", f"{total_audio_sec:.1f}s ({total_audio_sec/60:.1f} min)")
            kcol2.metric("Throughput", f"{throughput:.2f} audio-sec/wall-sec")
            kcol3.metric("Avg RTFx", f"{avg_rtfx:.0f}x real-time")
            kcol4.metric("Avg Latency", f"{total_proc_sec/len(actual_results):.2f}s")
            kcol5.metric("Success Rate", f"{len(actual_results)}/{len(actual_results)+len(failed_results)}")
            
            # Cost projection
            st.subheader("💰 Cost Projection")
            avg_cost_spot = sum(r["cost_spot"] for r in actual_results) / len(actual_results)
            daily_capacity_min = throughput * 60 * 24  # minutes/day
            
            ccol1, ccol2, ccol3 = st.columns(3)
            ccol1.metric("Cost (Spot) / 1K min", f"${avg_cost_spot:.2f}")
            ccol2.metric("Daily Capacity", f"{daily_capacity_min:.0f} min/day")
            ccol3.metric("Daily Cost (24/7 Spot)", f"${0.42 * 24:.2f}")
            
            # Per-tier breakdown
            st.subheader("🎯 Per-Tier Performance")
            tier_data = {}
            for r in actual_results:
                tier = r["engine"]
                if tier not in tier_data:
                    tier_data[tier] = []
                tier_data[tier].append(r)
            
            tier_cols = st.columns(len(tier_data))
            for idx, (tier, tier_results) in enumerate(sorted(tier_data.items())):
                with tier_cols[idx]:
                    avg_tier_rtfx = sum(r["rtfx"] for r in tier_results) / len(tier_results)
                    avg_tier_latency = sum(r["processing_sec"] for r in tier_results) / len(tier_results)
                    total_tier_audio = sum(r["duration_sec"] for r in tier_results)
                    st.markdown(f"**{tier.upper()}**")
                    st.write(f"Files: {len(tier_results)}")
                    st.write(f"Audio: {total_tier_audio:.1f}s")
                    st.write(f"Avg RTFx: {avg_tier_rtfx:.0f}x")
                    st.write(f"Avg Latency: {avg_tier_latency:.2f}s")
            
            # Detailed results table
            st.subheader("📋 Detailed Results")
            display_results = []
            for r in actual_results:
                display_results.append({
                    "File": r["file"],
                    "Language": r["language"],
                    "Engine": r["engine"],
                    "Audio (s)": f"{r['duration_sec']:.1f}",
                    "Processing (s)": f"{r['processing_sec']:.2f}",
                    "RTFx": f"{r['rtfx']:.0f}x",
                    "Lang Det. (ms)": f"{r['lang_detection_ms']:.0f}",
                    "Transcription (s)": f"{r['transcription_sec']:.2f}",
                    "Cost/1K min": f"${r['cost_spot']:.2f}"
                })
            
            st.dataframe(display_results, use_container_width=True, hide_index=True)
            
            # Download results
            import io
            import csv
            
            csv_buffer = io.StringIO()
            if display_results:
                writer = csv.DictWriter(csv_buffer, fieldnames=display_results[0].keys())
                writer.writeheader()
                writer.writerows(display_results)
            
            st.download_button(
                "⬇️ Download Benchmark CSV",
                data=csv_buffer.getvalue(),
                file_name=f"blitz_benchmark_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

# =============================================================================
# TAB 5: System Monitor (Live Metrics from /metrics endpoint)
# =============================================================================
with tab5:
    st.header("🖥️ System Monitor")
    st.markdown("Live metrics from the Blitz observability layer (`/metrics` endpoint).")

    def parse_prometheus_metrics(text):
        """Parse Prometheus text format into a dict of metric_name -> list of {labels, value}."""
        metrics = {}
        for line in text.strip().split("\n"):
            if line.startswith("#") or not line.strip():
                continue
            match = re.match(r'([a-zA-Z_:][a-zA-Z0-9_:]*)(\{.*?\})?\s+([\d.e+\-NnAa]+)', line)
            if match:
                name, labels_str, value_str = match.groups()
                try:
                    value = float(value_str)
                except ValueError:
                    continue
                labels = {}
                if labels_str:
                    label_matches = re.findall(r'(\w+)="([^"]*)"', labels_str)
                    labels = dict(label_matches)
                metrics.setdefault(name, []).append({"labels": labels, "value": value})
        return metrics

    def get_metric_value(metrics, name, labels=None):
        """Get a specific metric value, optionally filtered by labels."""
        entries = metrics.get(name, [])
        if not entries:
            return None
        if labels:
            for entry in entries:
                if all(entry["labels"].get(k) == v for k, v in labels.items()):
                    return entry["value"]
            return None
        return entries[0]["value"] if entries else None

    def get_metric_sum(metrics, name):
        """Sum all values for a metric across all label combinations."""
        entries = metrics.get(name, [])
        return sum(e["value"] for e in entries)

    # Auto-refresh
    auto_refresh = st.checkbox("Auto-refresh (every 5s)", value=False)
    if auto_refresh:
        time.sleep(5)
        st.rerun()

    col_refresh, col_status = st.columns([1, 3])
    with col_refresh:
        if st.button("🔄 Refresh Metrics", use_container_width=True):
            st.rerun()

    try:
        metrics_response = requests.get(f"{BLITZ_API}/metrics", timeout=5)
        metrics = parse_prometheus_metrics(metrics_response.text)

        with col_status:
            st.success("🟢 Metrics endpoint reachable")

        # ---- GPU Section ----
        st.subheader("🎮 GPU Status")
        gpu_mem_used = get_metric_value(metrics, "blitz_gpu_memory_used_bytes", {"gpu_id": "0"})
        gpu_mem_total = get_metric_value(metrics, "blitz_gpu_memory_total_bytes", {"gpu_id": "0"})
        gpu_util = get_metric_value(metrics, "blitz_gpu_utilization_percent", {"gpu_id": "0"})

        gcol1, gcol2, gcol3 = st.columns(3)
        if gpu_mem_used and gpu_mem_total:
            used_gb = gpu_mem_used / (1024**3)
            total_gb = gpu_mem_total / (1024**3)
            gcol1.metric("VRAM Used", f"{used_gb:.1f} GB / {total_gb:.1f} GB")
            gcol2.metric("VRAM Utilization", f"{(used_gb/total_gb)*100:.0f}%")
        else:
            gcol1.metric("VRAM Used", "N/A (waiting for refresh)")
            gcol2.metric("VRAM Utilization", "N/A")

        if gpu_util is not None:
            gcol3.metric("GPU Compute", f"{gpu_util:.0f}%")
        else:
            gcol3.metric("GPU Compute", "N/A")

        # ---- Request Metrics ----
        st.subheader("📈 Request Metrics")
        
        total_requests = get_metric_sum(metrics, "blitz_requests_total")
        in_progress = get_metric_value(metrics, "blitz_requests_in_progress")
        transcribe_reqs = get_metric_value(metrics, "blitz_requests_total", 
                                           {"endpoint": "/transcribe", "method": "POST", "status_code": "200"}) or 0
        errors = get_metric_sum(metrics, "blitz_transcription_errors_total")

        rcol1, rcol2, rcol3, rcol4 = st.columns(4)
        rcol1.metric("Total Requests", f"{total_requests:.0f}")
        rcol2.metric("Transcriptions", f"{transcribe_reqs:.0f}")
        rcol3.metric("In Progress", f"{in_progress:.0f}" if in_progress else "0")
        rcol4.metric("Errors", f"{errors:.0f}")

        # ---- Tier Routing ----
        st.subheader("🎯 Tier Routing")
        tier_entries = metrics.get("blitz_tier_routing_total", [])
        if tier_entries:
            tcol1, tcol2, tcol3 = st.columns(3)
            for entry in tier_entries:
                tier = entry["labels"].get("tier", "unknown")
                if tier == "tier1":
                    tcol1.metric("Tier 1 (English)", f"{entry['value']:.0f}")
                elif tier == "tier2":
                    tcol2.metric("Tier 2 (EU langs)", f"{entry['value']:.0f}")
                elif tier == "tier3":
                    tcol3.metric("Tier 3 (Whisper)", f"{entry['value']:.0f}")
        else:
            st.info("No tier routing data yet — transcribe some files first.")

        # ---- Audio Processed ----
        st.subheader("🎵 Audio Processed")
        audio_entries = metrics.get("blitz_audio_minutes_processed_total", [])
        if audio_entries:
            total_minutes = sum(e["value"] for e in audio_entries)
            st.metric("Total Audio Processed", f"{total_minutes:.2f} minutes")
        else:
            st.info("No audio processed yet.")

        # ---- Process Info ----
        with st.expander("🔧 Process Info"):
            cpu_seconds = get_metric_value(metrics, "process_cpu_seconds_total")
            mem_bytes = get_metric_value(metrics, "process_resident_memory_bytes")
            
            pcol1, pcol2 = st.columns(2)
            if cpu_seconds:
                pcol1.metric("CPU Time", f"{cpu_seconds:.1f}s")
            if mem_bytes:
                pcol2.metric("RAM Usage", f"{mem_bytes/(1024**3):.2f} GB")

        # ---- Raw Metrics (expandable) ----
        with st.expander("📄 Raw Metrics (Prometheus format)"):
            blitz_lines = [l for l in metrics_response.text.split("\n") if "blitz_" in l]
            st.code("\n".join(blitz_lines), language=None)

    except requests.exceptions.ConnectionError:
        with col_status:
            st.error("🔴 Cannot reach metrics endpoint")
    except Exception as e:
        st.error(f"Error fetching metrics: {str(e)}")


