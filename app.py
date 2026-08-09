import streamlit as st
import streamlit.components.v1 as components
import os
import io
import json
import tempfile
import ffmpeg
import requests
import base64
from datetime import timedelta
from st_click_detector import click_detector

# ==========================================
# Application Configuration
# ==========================================
st.set_page_config(page_title="Advanced STT - Human-in-the-Loop", layout="wide")

st.title("🎙️ מערכת תמלול מתקדמת - Human-in-the-Loop")
st.markdown("תמלול, זיהוי דוברים, ותיקון שגיאות אינטראקטיבי מבוסס Deepgram.")

# ==========================================
# State Management
# ==========================================
if "words_data" not in st.session_state:
    st.session_state.words_data = []
if "audio_bytes" not in st.session_state:
    st.session_state.audio_bytes = None
if "current_file_name" not in st.session_state:
    st.session_state.current_file_name = None
if "active_word_id" not in st.session_state:
    st.session_state.active_word_id = "" 
if "speaker_names" not in st.session_state:
    st.session_state.speaker_names = {} # שומר מיפוי בין ID דובר לשם אמיתי

# מילות חלל נפוצות לסינון ויזואלי
FILLERS = ["אה", "אממ", "אהה", "אמ", "um", "uh", "mhm", "mm", "ah", "er", "hmm"]

# ==========================================
# Helper Functions
# ==========================================
@st.cache_data(show_spinner=False)
def process_audio_cached(api_key, audio_bytes, language_choice):
    try:
        url = "https://api.deepgram.com/v1/listen"
        
        if language_choice == "he":
            params = {
                "model": "general",
                "tier": "nova-3", 
                "language": "he",
                "smart_format": "true",
                "punctuate": "true",
                "filler_words": "true",
                "words": "true",
                "diarize": "true"
            }
        else: 
            params = {
                "model": "nova-2",
                "language": "en",
                "smart_format": "true",
                "punctuate": "true",
                "filler_words": "true",
                "words": "true",
                "diarize": "true"
            }
        
        headers = {
            "Authorization": f"Token {api_key}",
            "Content-Type": "audio/*"
        }
        
        response = requests.post(url, headers=headers, params=params, data=audio_bytes)
        
        if response.status_code != 200:
            return {"error": f"API Error {response.status_code}: {response.text}"}
            
        data = response.json()
        
        if "results" not in data or not data["results"].get("channels"):
            return []
            
        channels = data["results"]["channels"][0]
        
        if not channels.get("alternatives"):
             return []
             
        alternatives = channels["alternatives"]
        primary_words = alternatives[0].get("words", [])
        
        words_list = []
        
        for i, word_obj in enumerate(primary_words):
            word_alts = set()
            for alt in alternatives[1:]:
                words_in_alt = alt.get("words", [])
                if i < len(words_in_alt):
                    if words_in_alt[i]["word"] != word_obj["word"]:
                        word_alts.add(words_in_alt[i]["word"])
            
            speaker = word_obj.get("speaker", 0)
            final_word = word_obj.get("punctuated_word", word_obj["word"])
            clean_word = word_obj["word"] 
            
            words_list.append({
                "id": i,
                "word": final_word,
                "original_word": final_word,
                "clean_word": clean_word, 
                "start": word_obj["start"],
                "end": word_obj["end"],
                "confidence": word_obj["confidence"],
                "speaker": speaker,
                "alternatives": list(word_alts),
                "deleted": False
            })
            
        return words_list
    except Exception as e:
        return {"error": str(e)}

def get_word_color(confidence):
    if confidence < 0.5: return "#ff4b4b" 
    elif confidence <= 0.9: return "#ffe14b" 
    else: return "#f0f2f6" 

def slice_audio(audio_bytes, start_sec, end_sec, padding=1.5):
    try:
        start_time = max(0, start_sec - padding)
        duration = (end_sec + padding) - start_time
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_in, \
             tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_out:
            
            temp_in.write(audio_bytes)
            temp_in.flush()
            
            in_file_path = temp_in.name
            out_file_path = temp_out.name

        try:
            (
                ffmpeg
                .input(in_file_path, ss=start_time)
                .output(out_file_path, t=duration, format='wav', acodec='pcm_s16le')
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            
            with open(out_file_path, "rb") as f:
                sliced_bytes = f.read()
                
            return sliced_bytes
        finally:
            if os.path.exists(in_file_path): os.remove(in_file_path)
            if os.path.exists(out_file_path): os.remove(out_file_path)
    except Exception as e:
        return None

def generate_txt(words_data, include_speakers=False):
    active_words = [w for w in words_data if not w.get("deleted", False)]
    if not include_speakers:
        return " ".join([w["word"] for w in active_words])
        
    txt = ""
    current_speaker = None
    for w in active_words:
        if w["speaker"] != current_speaker:
            spk_name = st.session_state.speaker_names.get(w["speaker"], f"דובר {w['speaker']}")
            txt += f"\n\n[{spk_name}]: "
            current_speaker = w["speaker"]
        txt += w["word"] + " "
    return txt.strip()

def generate_srt(words_data):
    active_words = [w for w in words_data if not w.get("deleted", False)]
    srt_content = ""
    chunk_index = 1
    chunk_words = []
    
    for i, word in enumerate(active_words):
        chunk_words.append(word)
        speaker_changed = i < len(active_words) - 1 and active_words[i+1]['speaker'] != word['speaker']
        
        if len(chunk_words) >= 10 or speaker_changed or i == len(active_words) - 1:
            start_time = timedelta(seconds=chunk_words[0]['start'])
            end_time = timedelta(seconds=chunk_words[-1]['end'])
            
            def format_time(td):
                total_sec = int(td.total_seconds())
                hours = total_sec // 3600
                minutes = (total_sec % 3600) // 60
                seconds = total_sec % 60
                millisecs = int((td.total_seconds() - total_sec) * 1000)
                return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millisecs:03d}"
            
            spk = chunk_words[0]['speaker']
            spk_name = st.session_state.speaker_names.get(spk, f"דובר {spk}")
            speaker_label = f"[{spk_name}] "
            text = " ".join([w['word'] for w in chunk_words])
            
            srt_content += f"{chunk_index}\n"
            srt_content += f"{format_time(start_time)} --> {format_time(end_time)}\n"
            srt_content += f"{speaker_label}{text}\n\n"
            
            chunk_index += 1
            chunk_words = []
            
    return srt_content

def call_gemini_api(api_key, text, prompt_type):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    
    if prompt_type == "סיכום כללי":
        prompt = f"סכם את התמלול הבא בצורה ברורה ותמציתית בעברית:\n\n{text}"
    elif prompt_type == "נקודות מפתח (Bullet points)":
        prompt = f"חלץ את נקודות המפתח והנושאים המרכזיים שעלו בתמלול הבא והצג אותם כרשימה בעברית:\n\n{text}"
    else:
        prompt = f"עשה ניתוח קצר של התמלול הבא (אווירה, מסקנות, רעיונות מרכזיים) בעברית:\n\n{text}"
        
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        resp = requests.post(url, headers=headers, json=data)
        resp.raise_for_status()
        return resp.json()['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        return f"שגיאה בתקשורת עם Gemini: {str(e)}"

# ==========================================
# UI Layout
# ==========================================

with st.sidebar:
    st.header("הגדרות")
    
    secret_key = st.secrets.get("DEEPGRAM_API_KEY", "")
    if not secret_key:
        api_key = st.text_input("Deepgram API Key", type="password")
        st.caption("השג מפתח חינמי באתר console.deepgram.com")
    else:
        api_key = secret_key
        st.success("✅ מחובר ל-Deepgram")
    
    st.divider()
    st.subheader("הגדרות תצוגה")
    gap_threshold = st.slider("התרעת השמטה (שניות)", min_value=0.3, max_value=2.0, value=0.6, step=0.1)
    
    if st.session_state.words_data:
        st.divider()
        with st.expander("👥 ניהול שמות דוברים"):
            active_spks = sorted(list(set(w["speaker"] for w in st.session_state.words_data if not w.get("deleted", False))))
            for spk in active_spks:
                current_name = st.session_state.speaker_names.get(spk, f"דובר {spk}")
                new_name = st.text_input(f"דובר {spk}:", value=current_name, key=f"spk_rename_{spk}")
                st.session_state.speaker_names[spk] = new_name
                
        with st.expander("🔍 חיפוש והחלפה גורפת"):
            search_t = st.text_input("חפש מילה/ביטוי:")
            replace_t = st.text_input("החלף ב:")
            if st.button("החלף הכל", use_container_width=True) and search_t:
                count = 0
                for w in st.session_state.words_data:
                    if not w.get("deleted") and (w["word"] == search_t or w["clean_word"] == search_t):
                        w["word"] = replace_t
                        w["confidence"] = 1.0
                        count += 1
                if count > 0:
                    st.success(f"הוחלפו {count} מופעים!")
                    st.rerun()
                else:
                    st.warning("המילה לא נמצאה.")

    st.divider()
    if st.button("איפוס מערכת (Clear Data)"):
        st.session_state.words_data = []
        st.session_state.audio_bytes = None
        st.session_state.current_file_name = None
        st.session_state.active_word_id = ""
        st.session_state.speaker_names = {}
        st.cache_data.clear()
        st.rerun()

st.subheader("1. הזנת אודיו")
col_upload, col_lang = st.columns([7, 3])

with col_upload:
    uploaded_file = st.file_uploader("העלה קובץ אודיו (MP3, WAV, M4A, OGG)", type=["mp3", "wav", "m4a", "ogg"])

with col_lang:
    st.write("בחר שפה:")
    language_choice = st.radio("שפה", ["he", "en"], format_func=lambda x: "עברית" if x == "he" else "אנגלית", label_visibility="collapsed")

if uploaded_file and (st.session_state.current_file_name != uploaded_file.name):
    st.session_state.words_data = []
    st.session_state.audio_bytes = uploaded_file.read()
    st.session_state.current_file_name = uploaded_file.name
    st.session_state.active_word_id = ""
    st.session_state.speaker_names = {}

if st.button("תמלל עכשיו", type="primary") and st.session_state.audio_bytes:
    if not api_key:
        st.error("אנא הזן Deepgram API Key בסרגל הצד.")
    else:
        with st.spinner("מתמלל בעזרת Deepgram..."):
            result = process_audio_cached(api_key, st.session_state.audio_bytes, language_choice)
            if isinstance(result, dict) and "error" in result:
                 st.error(f"שגיאה בתמלול: {result['error']}")
            elif isinstance(result, list):
                if len(result) == 0:
                    st.warning("לא זוהו מילים.")
                else:
                    st.session_state.words_data = result
                    st.session_state.active_word_id = ""
                    st.success("התמלול עבר בהצלחה!")
                    st.rerun()

if st.session_state.words_data:
    st.divider()
    col_viz, col_edit = st.columns([6, 4])
    active_words = [w for w in st.session_state.words_data if not w.get("deleted", False)]
    
    with col_viz:
        st.subheader("2. תמלול (לחץ כדי לתקן)")
        with st.container(height=650):
            current_speaker = None
            html_text = "<div style='line-height: 2.5; font-size: 18px; direction: rtl; padding-bottom: 50px;'>"
            
            for i, w in enumerate(active_words):
                if w["speaker"] != current_speaker:
                    if current_speaker is not None: html_text += "<br><br>" 
                    current_speaker = w["speaker"]
                    spk_name = st.session_state.speaker_names.get(current_speaker, f"דובר {current_speaker}")
                    html_text += f"<strong style='color:#555;'>[{spk_name}]: </strong>"
                
                color = get_word_color(w["confidence"])
                is_filler = w.get("clean_word", "").lower().strip(",.?!") in FILLERS
                display_text = f"[{w['word']}]" if is_filler else w['word']
                is_active = (st.session_state.active_word_id == str(w['id']))
                
                border_style = "2px solid #2196F3" if is_active else "2px solid transparent"
                box_shadow = "0 0 8px rgba(33,150,243,0.8)" if is_active else "0 1px 2px rgba(0,0,0,0.1)"
                opacity_val = "1" if (not is_filler or is_active) else "0.5"
                font_start = "<i>" if is_filler else ""
                font_end = "</i>" if is_filler else ""
                
                span_style = f"background-color: {color}; padding: 4px 8px; border-radius: 6px; margin: 0 2px; transition: 0.2s; border: {border_style}; box-shadow: {box_shadow}; opacity: {opacity_val}; display: inline-block;"
                
                html_text += f"<a href='javascript:void(0);' id='{w['id']}' style='text-decoration: none; color: inherit;'>"
                html_text += f"<span style='{span_style}' onmouseover=\"this.style.opacity='0.8'\" onmouseout=\"this.style.opacity='{opacity_val}'\">{font_start}{display_text}{font_end}</span></a> "
                
                if i < len(active_words) - 1:
                    next_w = active_words[i+1]
                    if next_w["speaker"] == w["speaker"]:
                        gap = next_w["start"] - w["end"]
                        if gap >= gap_threshold:
                            html_text += f"<span style='color: #ff9800; font-size: 12px; margin: 0 4px;' title='שתיקה של {gap:.1f} שניות'>[⏳]</span> "
                
            html_text += "</div>"
            
            if st.session_state.active_word_id:
                html_text += f"""<img src="dummy" style="display:none;" onerror="setTimeout(function() {{ var el = document.getElementById('{st.session_state.active_word_id}'); if(el) el.scrollIntoView({{behavior: 'smooth', block: 'center'}}); }}, 300);">"""
            
            clicked_word_id = click_detector(html_text)
            if clicked_word_id and clicked_word_id != st.session_state.active_word_id:
                st.session_state.active_word_id = clicked_word_id
                st.rerun()
    
    with col_edit:
        st.subheader("3. ממשק תיקון")
        
        if st.session_state.active_word_id:
            selected_id = int(st.session_state.active_word_id)
            word_obj = next((w for w in st.session_state.words_data if w["id"] == selected_id), None)
            
            if word_obj:
                spk_name = st.session_state.speaker_names.get(word_obj['speaker'], f"דובר {word_obj['speaker']}")
                st.write(f"**מילה נבחרת:** `{word_obj['word']}` ({spk_name})")
                
                sliced_audio = slice_audio(st.session_state.audio_bytes, word_obj["start"], word_obj["end"], padding=1.5)
                if sliced_audio:
                    # נגן HTML חכם עם שליטה במהירות
                    b64 = base64.b64encode(sliced_audio).decode()
                    audio_html = f"""
                    <div style="text-align:center;">
                        <audio id="audio-player" controls style="width: 100%; height: 40px;">
                          <source src="data:audio/wav;base64,{b64}" type="audio/wav">
                        </audio>
                        <div style="margin-top: 5px; display: flex; justify-content: center; gap: 8px;">
                            <span style="font-size:12px; font-family:sans-serif; color: gray; align-self: center;">מהירות:</span>
                            <button onclick="document.getElementById('audio-player').playbackRate = 0.5" style="border:1px solid #ccc; border-radius:4px; cursor:pointer;">0.5x</button>
                            <button onclick="document.getElementById('audio-player').playbackRate = 0.75" style="border:1px solid #ccc; border-radius:4px; cursor:pointer;">0.75x</button>
                            <button onclick="document.getElementById('audio-player').playbackRate = 1.0" style="border:1px solid #ccc; border-radius:4px; cursor:pointer;">1.0x</button>
                        </div>
                    </div>
                    """
                    components.html(audio_html, height=85)
                
                alts = [word_obj['word']] + word_obj.get('alternatives', [])
                chosen_alt = st.selectbox("הצעות המודל:", alts)
                
                # טופס תיקון מילה - לחיצה על Enter תשמור ותעביר הלאה!
                with st.form("edit_word_form"):
                    col_text, col_speaker = st.columns([3, 1])
                    with col_text:
                        manual_text = st.text_input("תיקון ידני:", value=chosen_alt)
                    with col_speaker:
                        new_speaker = st.number_input("דובר:", min_value=0, max_value=20, value=int(word_obj['speaker']), step=1)
                    
                    col_b1, col_b2 = st.columns(2)
                    with col_b1:
                        submit_btn = st.form_submit_button("✅ שמור והמשך (Enter)", use_container_width=True)
                    with col_b2:
                        delete_btn = st.form_submit_button("🗑️ מחק מילה", use_container_width=True)
                        
                    if submit_btn:
                        real_idx = st.session_state.words_data.index(word_obj)
                        st.session_state.words_data[real_idx]["word"] = manual_text
                        st.session_state.words_data[real_idx]["speaker"] = new_speaker
                        st.session_state.words_data[real_idx]["confidence"] = 1.0 
                        # מעבר אוטומטי למילה הבאה
                        active_idx = active_words.index(word_obj)
                        if active_idx + 1 < len(active_words):
                            st.session_state.active_word_id = str(active_words[active_idx+1]["id"])
                        st.rerun()
                        
                    if delete_btn:
                        real_idx = st.session_state.words_data.index(word_obj)
                        st.session_state.words_data[real_idx]["deleted"] = True
                        active_idx = active_words.index(word_obj)
                        if active_idx + 1 < len(active_words):
                            st.session_state.active_word_id = str(active_words[active_idx+1]["id"])
                        else:
                            st.session_state.active_word_id = ""
                        st.rerun()
                
                st.divider()
                st.write("🛠️ **המערכת השמיטה מילה?**")
                new_word_text = st.text_input("המילה החסרה:")
                col_add1, col_add2 = st.columns(2)
                with col_add1:
                    if st.button("➕ הוסף לפני", use_container_width=True) and new_word_text:
                        new_id = max([w["id"] for w in st.session_state.words_data]) + 1
                        new_word = {"id": new_id, "word": new_word_text, "original_word": None, "clean_word": new_word_text, "start": max(0.0, word_obj["start"] - 0.1), "end": word_obj["start"], "confidence": 1.0, "speaker": word_obj["speaker"], "alternatives": [], "deleted": False}
                        idx = st.session_state.words_data.index(word_obj)
                        st.session_state.words_data.insert(idx, new_word)
                        st.session_state.active_word_id = str(new_id) 
                        st.rerun()
                with col_add2:
                    if st.button("➕ הוסף אחרי", use_container_width=True) and new_word_text:
                        new_id = max([w["id"] for w in st.session_state.words_data]) + 1
                        new_word = {"id": new_id, "word": new_word_text, "original_word": None, "clean_word": new_word_text, "start": word_obj["end"], "end": word_obj["end"] + 0.1, "confidence": 1.0, "speaker": word_obj["speaker"], "alternatives": [], "deleted": False}
                        idx = st.session_state.words_data.index(word_obj)
                        st.session_state.words_data.insert(idx + 1, new_word)
                        st.session_state.active_word_id = str(new_id) 
                        st.rerun()
        else:
            st.info("👈 לחץ על מילה כלשהי בטקסט מימין כדי לפתוח את ממשק התיקון שלה.")

    st.divider()
    st.subheader("4. ייצוא נתונים")
    col_ex1, col_ex2, col_ex3, col_ex4 = st.columns(4)
    
    final_text = generate_txt(st.session_state.words_data, include_speakers=False)
    final_text_with_speakers = generate_txt(st.session_state.words_data, include_speakers=True)
    
    original_text = " ".join([w["original_word"] for w in st.session_state.words_data if w.get("original_word") is not None])
    if not original_text: original_text = final_text
    
    with col_ex1:
        st.download_button("📝 טקסט מתוקן (TXT)", data=final_text_with_speakers, file_name="transcript.txt", mime="text/plain", use_container_width=True)
    with col_ex2:
        st.download_button("📜 טקסט מקורי (TXT)", data=original_text, file_name="original.txt", mime="text/plain", use_container_width=True)
    with col_ex3:
        st.download_button("🎬 כתוביות (SRT)", data=generate_srt(active_words), file_name="subtitles.srt", mime="text/plain", use_container_width=True)
    with col_ex4:
         st.download_button("⚙️ נתונים (JSON)", data=json.dumps(active_words, ensure_ascii=False, indent=2), file_name="data.json", mime="application/json", use_container_width=True)

    st.divider()
    st.subheader("5. ניתוח חכם (AI)")
    with st.expander("🤖 יצירת סיכום / נקודות מפתח באמצעות תמליל השיחה"):
        st.write("הזן מפתח API של Gemini כדי לנתח את הטקסט המתוקן (חינם ב-Google AI Studio).")
        gemini_key = st.text_input("Gemini API Key:", type="password")
        prompt_type = st.selectbox("מה תרצה להפיק מהתמלול?", ["סיכום כללי", "נקודות מפתח (Bullet points)", "ניתוח (אווירה ומסקנות)"])
        
        if st.button("🚀 נתח טקסט", type="primary"):
            if not gemini_key:
                st.error("יש להזין מפתח API של Gemini תחילה.")
            else:
                with st.spinner("ה-AI קורא ומנתח את התמלול..."):
                    ai_response = call_gemini_api(gemini_key, final_text_with_speakers, prompt_type)
                    st.markdown("### תוצאה:")
                    st.info(ai_response)
