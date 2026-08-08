import streamlit as st
import os
import io
import json
import tempfile
import ffmpeg
import requests
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
    st.session_state.active_word_id = "" # שומר את ה-ID של המילה הפעילה

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
                # הסרנו את "alternatives": 3 כי nova-2 לא תומך בזה יותר
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
            # אם יש חלופות (במודלים ישנים או בעברית), נוסיף אותן. באנגלית זה פשוט ידלג.
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
    if confidence < 0.5:
        return "#ff4b4b" 
    elif confidence <= 0.9:
        return "#ffe14b" 
    else:
        return "#f0f2f6" 

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
            if os.path.exists(in_file_path):
                os.remove(in_file_path)
            if os.path.exists(out_file_path):
                os.remove(out_file_path)
                
    except ffmpeg.Error as e:
        return None
    except Exception as e:
        return None

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
            
            speaker_label = f"[דובר {chunk_words[0]['speaker']}] "
            text = " ".join([w['word'] for w in chunk_words])
            
            srt_content += f"{chunk_index}\n"
            srt_content += f"{format_time(start_time)} --> {format_time(end_time)}\n"
            srt_content += f"{speaker_label}{text}\n\n"
            
            chunk_index += 1
            chunk_words = []
            
    return srt_content

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
        st.success("✅ מחובר לשרתי התמלול")
    
    st.divider()
    st.subheader("הגדרות תצוגה")
    gap_threshold = st.slider("התרעת השמטה (שניות)", min_value=0.3, max_value=2.0, value=0.6, step=0.1, 
                              help="פער זמנים בין מילים שמעליו המערכת תציג סמל [⏳] שמתריע על שתיקה ארוכה או מילה חסרה.")
    
    st.divider()
    if st.button("איפוס מערכת (Clear Data)"):
        st.session_state.words_data = []
        st.session_state.audio_bytes = None
        st.session_state.current_file_name = None
        st.session_state.active_word_id = ""
        st.cache_data.clear()
        st.rerun()

st.subheader("1. הזנת אודיו")
col_upload, col_lang = st.columns([7, 3])

with col_upload:
    uploaded_file = st.file_uploader("העלה קובץ אודיו (MP3, WAV, M4A, OGG)", type=["mp3", "wav", "m4a", "ogg"])

with col_lang:
    st.write("בחר את השפה המרכזית באודיו:")
    language_choice = st.radio(
        "שפה",
        options=["he", "en"],
        format_func=lambda x: "עברית (Hebrew)" if x == "he" else "אנגלית (English)",
        label_visibility="collapsed"
    )

if uploaded_file and (st.session_state.current_file_name != uploaded_file.name):
    st.session_state.words_data = []
    st.session_state.audio_bytes = uploaded_file.read()
    st.session_state.current_file_name = uploaded_file.name
    st.session_state.active_word_id = ""

if st.button("תמלל עכשיו", type="primary") and st.session_state.audio_bytes:
    if not api_key:
        st.error("אנא הזן Deepgram API Key בסרגל הצד.")
    else:
        with st.spinner("מתמלל בעזרת Deepgram... (זה עשוי לקחת כמה שניות)"):
            result = process_audio_cached(api_key, st.session_state.audio_bytes, language_choice)
            
            if isinstance(result, dict) and "error" in result:
                 st.error(f"שגיאה בתמלול: {result['error']}")
            elif isinstance(result, list):
                if len(result) == 0:
                    st.warning("התמלול הסתיים, אבל המודל לא זיהה שום מילים ברורות באודיו הזה.")
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
        st.subheader("2. תמלול (לחץ על מילה כדי לתקן)")
        
        with st.container(height=650):
            current_speaker = None
            html_text = "<div style='line-height: 2.5; font-size: 18px; direction: rtl; padding-bottom: 50px;'>"
            
            for i, w in enumerate(active_words):
                if w["speaker"] != current_speaker:
                    if current_speaker is not None:
                        html_text += "<br><br>" 
                    current_speaker = w["speaker"]
                    html_text += f"<strong style='color:#555;'>[דובר {current_speaker}]: </strong>"
                
                color = get_word_color(w["confidence"])
                is_filler = w.get("clean_word", "").lower().strip(",.?!") in FILLERS
                display_text = f"[{w['word']}]" if is_filler else w['word']
                
                # בדיקה האם זו המילה הלחוצה כרגע
                is_active = (st.session_state.active_word_id == str(w['id']))
                
                # יצירת הסטייל של המילה (כולל מסגרת כחולה אם היא פעילה)
                border_style = "2px solid #2196F3" if is_active else "2px solid transparent"
                box_shadow = "0 0 8px rgba(33,150,243,0.8)" if is_active else "0 1px 2px rgba(0,0,0,0.1)"
                opacity_val = "1" if (not is_filler or is_active) else "0.5"
                font_start = "<i>" if is_filler else ""
                font_end = "</i>" if is_filler else ""
                
                span_style = f"background-color: {color}; padding: 4px 8px; border-radius: 6px; margin: 0 2px; transition: 0.2s; border: {border_style}; box-shadow: {box_shadow}; opacity: {opacity_val}; display: inline-block;"
                
                html_text += f"<a href='javascript:void(0);' id='{w['id']}' style='text-decoration: none; color: inherit;'>"
                html_text += f"<span style='{span_style}' onmouseover=\"this.style.opacity='0.8'\" onmouseout=\"this.style.opacity='{opacity_val}'\">{font_start}{display_text}{font_end}</span>"
                html_text += "</a> "
                
                if i < len(active_words) - 1:
                    next_w = active_words[i+1]
                    if next_w["speaker"] == w["speaker"]:
                        gap = next_w["start"] - w["end"]
                        if gap >= gap_threshold:
                            html_text += f"<span style='color: #ff9800; font-size: 12px; margin: 0 4px; cursor: help;' title='שתיקה של {gap:.1f} שניות - ייתכן שהושמטה מילה'>[⏳]</span> "
                
            html_text += "</div>"
            
            # הזרקת קוד JS שגולל אוטומטית למילה הפעילה (עוקף את חסימת ה-scripts)
            if st.session_state.active_word_id:
                html_text += f"""
                <img src="dummy_url" style="display:none;" onerror="
                    setTimeout(function() {{
                        var el = document.getElementById('{st.session_state.active_word_id}');
                        if(el) {{
                            el.scrollIntoView({{behavior: 'smooth', block: 'center'}});
                        }}
                    }}, 300);
                ">
                """
            
            clicked_word_id = click_detector(html_text)
            
            # אם נלחצה מילה חדשה, נעדכן את הסטייט ונרענן כדי להחיל עליה את העיצוב
            if clicked_word_id and clicked_word_id != st.session_state.active_word_id:
                st.session_state.active_word_id = clicked_word_id
                st.rerun()
    
    with col_edit:
        st.subheader("3. ממשק תיקון")
        
        if st.session_state.active_word_id:
            selected_id = int(st.session_state.active_word_id)
            word_obj = next((w for w in st.session_state.words_data if w["id"] == selected_id), None)
            
            if word_obj:
                st.write(f"**מילה נבחרת:** `{word_obj['word']}` (דובר {word_obj['speaker']})")
                
                sliced_audio = slice_audio(st.session_state.audio_bytes, word_obj["start"], word_obj["end"], padding=1.5)
                if sliced_audio:
                    st.audio(sliced_audio, format="audio/wav")
                else:
                    st.warning("לא ניתן לנגן אודיו חתוך (האם FFmpeg מותקן?)")
                
                alts = [word_obj['word']] + word_obj['alternatives']
                chosen_alt = st.selectbox("הצעות המודל:", alts)
                
                col_text, col_speaker = st.columns([3, 1])
                with col_text:
                    manual_text = st.text_input("תיקון ידני:", value=chosen_alt)
                with col_speaker:
                    new_speaker = st.number_input("דובר:", min_value=0, max_value=20, value=int(word_obj['speaker']), step=1)
                
                col_btn1, col_btn2 = st.columns(2)
                
                with col_btn1:
                    if st.button("✅ שמור תיקון", use_container_width=True):
                        st.session_state.words_data[selected_id]["word"] = manual_text
                        st.session_state.words_data[selected_id]["speaker"] = new_speaker
                        st.session_state.words_data[selected_id]["confidence"] = 1.0 
                        st.rerun()
                
                with col_btn2:
                    if st.button("🗑️ מחק מילה", use_container_width=True):
                        st.session_state.words_data[selected_id]["deleted"] = True
                        st.session_state.active_word_id = "" # מנקה את הבחירה
                        st.rerun()
                
                st.divider()
                st.write("🛠️ **המערכת השמיטה מילה?**")
                st.caption("הקלד את המילה החסרה ולחץ איפה למקם אותה ביחס למילה שבחרת.")
                new_word_text = st.text_input("המילה החסרה:")
                
                col_add1, col_add2 = st.columns(2)
                with col_add1:
                    if st.button("➕ הוסף לפני", use_container_width=True) and new_word_text:
                        new_id = max([w["id"] for w in st.session_state.words_data]) + 1
                        new_word = {
                            "id": new_id,
                            "word": new_word_text,
                            "original_word": None,
                            "clean_word": new_word_text,
                            "start": max(0.0, word_obj["start"] - 0.1),
                            "end": word_obj["start"],
                            "confidence": 1.0,
                            "speaker": word_obj["speaker"],
                            "alternatives": [],
                            "deleted": False
                        }
                        idx = st.session_state.words_data.index(word_obj)
                        st.session_state.words_data.insert(idx, new_word)
                        st.session_state.active_word_id = str(new_id) # מסמן את המילה החדשה מיד
                        st.rerun()
                        
                with col_add2:
                    if st.button("➕ הוסף אחרי", use_container_width=True) and new_word_text:
                        new_id = max([w["id"] for w in st.session_state.words_data]) + 1
                        new_word = {
                            "id": new_id,
                            "word": new_word_text,
                            "original_word": None,
                            "clean_word": new_word_text,
                            "start": word_obj["end"],
                            "end": word_obj["end"] + 0.1,
                            "confidence": 1.0,
                            "speaker": word_obj["speaker"],
                            "alternatives": [],
                            "deleted": False
                        }
                        idx = st.session_state.words_data.index(word_obj)
                        st.session_state.words_data.insert(idx + 1, new_word)
                        st.session_state.active_word_id = str(new_id) # מסמן את המילה החדשה מיד
                        st.rerun()
        else:
            st.info("👈 לחץ על מילה כלשהי בטקסט מימין כדי לפתוח את ממשק התיקון שלה.")

    st.divider()
    st.subheader("4. ייצוא נתונים")
    
    col_ex1, col_ex2, col_ex3, col_ex4 = st.columns(4)
    
    final_text = " ".join([w["word"] for w in active_words])
    
    # שחזור הטקסט המקורי (מתעלם ממילים שנוספו ידנית)
    original_text = " ".join([w["original_word"] for w in st.session_state.words_data if w.get("original_word") is not None])
    if not original_text:
        original_text = final_text
    
    with col_ex1:
        st.download_button(
            label="📝 טקסט מתוקן (TXT)",
            data=final_text,
            file_name="transcript_edited.txt",
            mime="text/plain",
            use_container_width=True,
            help="מוריד את הטקסט הסופי לאחר כל התיקונים, המחיקות והתוספות שלך."
        )
        
    with col_ex2:
        st.download_button(
            label="📜 טקסט מקורי (TXT)",
            data=original_text,
            file_name="transcript_original.txt",
            mime="text/plain",
            use_container_width=True,
            help="מוריד את התמלול הגולמי כפי שהתקבל במקור מהמודל, לפני שביצעת בו שינויים."
        )
    
    with col_ex3:
        srt_data = generate_srt(active_words)
        st.download_button(
            label="🎬 כתוביות (SRT)",
            data=srt_data,
            file_name="subtitles.srt",
            mime="text/plain",
            use_container_width=True,
            help="קובץ כתוביות הכולל תזמונים מדויקים וחלוקה לדוברים. מיועד לעריכת וידאו."
        )
        
    with col_ex4:
         json_data = json.dumps(active_words, ensure_ascii=False, indent=2)
         st.download_button(
            label="⚙️ נתונים (JSON)",
            data=json_data,
            file_name="data.json",
            mime="application/json",
            use_container_width=True,
            help="קובץ הגיבוי הטכני של כלל מאגר הנתונים באפליקציה (למפתחים)."
        )
