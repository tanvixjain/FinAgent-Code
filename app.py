import os
import logging
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Ensure working directory is the script directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Optional Gemini / Config
try:
    import config
    GEMINI_API_KEY_FROM_CONFIG = getattr(config, 'GEMINI_API_KEY', "")
except ImportError:
    GEMINI_API_KEY_FROM_CONFIG = ""

try:
    from langdetect import detect as detect_lang
except Exception:
    detect_lang = None

try:
    from googletrans import Translator as GoogleTranslator
except Exception:
    GoogleTranslator = None

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FinAIgent")

# --- Flask app ---
app = Flask(__name__)
CORS(app)

# ----------------------
# Gemini / GenAI config
# Talks to the Gemini REST API directly with `requests` (no google.generativeai / grpc).
# ----------------------
import requests

GEMINI_API_KEY = GEMINI_API_KEY_FROM_CONFIG or os.getenv("GEMINI_API_KEY", "")
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"


class _GeminiResponse:
    def __init__(self, text):
        self.text = text


class GeminiRestModel:
    """Minimal stand-in for genai.GenerativeModel: generate_content(prompt) -> object with .text"""

    def __init__(self, model_name, api_key, timeout=60):
        self.model_name = model_name if model_name.startswith("models/") else f"models/{model_name}"
        self.api_key = api_key
        self.timeout = timeout

    def generate_content(self, prompt):
        resp = requests.post(
            f"{GEMINI_API_BASE}/{self.model_name}:generateContent",
            headers={"x-goog-api-key": self.api_key},
            json={"contents": [{"parts": [{"text": str(prompt)}]}]},
            timeout=self.timeout,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Gemini API error {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        try:
            parts = data["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts)
        except (KeyError, IndexError, TypeError):
            raise ValueError(f"Gemini returned no text: {str(data)[:300]}")
        return _GeminiResponse(text)


gemini_model = None
if GEMINI_API_KEY:
    model_candidates = [
        "models/gemini-3.5-flash-lite",
        "models/gemini-flash-latest",
    ]
    for m in model_candidates:
        try:
            candidate = GeminiRestModel(m, GEMINI_API_KEY, timeout=30)
            _ = candidate.generate_content("ping")
            candidate.timeout = 60
            gemini_model = candidate
            logger.info("Initialized and verified Gemini model: %s", m)
            break
        except Exception as e:
            logger.warning("Gemini model %s not usable: %s", m, e)
    if gemini_model is None:
        logger.warning("No Gemini model could be initialized or verified; running in RAG-only mode.")
else:
    logger.info("No GEMINI_API_KEY set; running retrieval-only fallback (no LLM).")

# ----------------------
# Translation setup
# ----------------------
translator = None
if GoogleTranslator is not None:
    try:
        # Use a more stable initialization for googletrans
        translator = GoogleTranslator()
        # Test translation
        _ = translator.translate("test", dest="hi")
        logger.info("googletrans translator initialized and verified.")
    except Exception as e:
        translator = None
        logger.warning("Failed to initialize or verify googletrans: %s. Translations will be limited.", e)
else:
    logger.info("googletrans not installed; translations disabled.")

if detect_lang is None:
    logger.info("langdetect not installed; language auto-detection disabled.")
else:
    logger.info("langdetect available for language detection.")


def detect_language_of_text(text: str, default: str = 'en') -> str:
    """
    Return detected language code for text if langdetect present; else default.
    """
    if not text or not isinstance(text, str) or not text.strip():
        return default
    if detect_lang is None:
        return default
    try:
        lang = detect_lang(text)
        return lang or default
    except Exception as e:
        logger.debug("Language detection failed: %s", e)
        return default


def translate_text(text: str, src: str = 'auto', dest: str = 'en'):
    """
    Translate using googletrans if available; fallback to Gemini if not.
    """
    if not text:
        return text, src
    
    # Try googletrans first if available
    if translator is not None:
        try:
            res = translator.translate(text, src=src, dest=dest)
            translated = getattr(res, 'text', str(res))
            detected = getattr(res, 'src', src)
            return translated, detected
        except Exception as e:
            logger.debug("googletrans failed: %s", e)

    # Fallback to Gemini for translation
    if gemini_model is not None:
        try:
            logger.info("Falling back to Gemini for translation to %s", dest)
            prompt = f"Translate the following text to {dest}. Only return the translated text without ANY commentary.\n\nText: {text}"
            response = gemini_model.generate_content(prompt)
            # Safe way to get text from Gemini
            try:
                res_text = response.text
                if res_text:
                    logger.info("Gemini translation successful.")
                    return res_text.strip(), src
            except (AttributeError, ValueError):
                logger.warning("Gemini translation response had no text.")
        except Exception as e:
            logger.error("Gemini translation fallback failed: %s", e)

    return text, src

# ----------------------
# Datasets configuration
# ----------------------
datasets_info = [
    {
        "topic": "Scam Awareness and Prevention",
        "path": "Scam_Awareness___Prevention_Q_A_Dataset_NoRepeatedQuestions.csv",
        "q_col": "user_question",
        "display_cols": ["detailed_answer"],
    },
    {
        "topic": "Fraud Scam Types",
        "path": "fraud_scam_type.csv",
        "q_col": "scam_example_text",
        "display_cols": ["scam_type", "explanation", "recommended_action"],
    },
    {
        "topic": "Document Corpus Extended",
        "path": "Document_corpus_extended.csv",
        "q_col": "title",
        "display_cols": ["chunk_text"],
    },
    {
        "topic": "Personal Budgeting and Finance",
        "path": "Budgeting_Personal_Finance_QA.csv",
        "q_col": "user_question",
        "display_cols": ["detailed_answer", "category"],
    },
    {
        "topic": "Government Schemes and Banking",
        "path": "Government_Schemes_Banking.csv",
        "q_col": "scheme_name",
        "display_cols": ["description", "eligibility", "benefits", "how_to_apply", "category"],
    },
]

models = {}

# ----------------------
# Train / load retrieval models
# ----------------------
for info in datasets_info:
    topic = info["topic"]
    path = info["path"]
    q_col = info["q_col"]
    display_cols = info["display_cols"]
    try:
        if not os.path.isfile(path):
            logger.warning("File not found, skipping dataset: %s", path)
            continue
        df = pd.read_csv(path)
        required_cols = [q_col] + display_cols
        if not all(col in df.columns for col in required_cols):
            logger.warning("Dataset %s missing required columns %s - skipping", path, required_cols)
            continue
        df = df[required_cols].copy()
        df.dropna(subset=[q_col], inplace=True)
        df.reset_index(drop=True, inplace=True)
        if df.empty:
            logger.warning("Dataset %s empty after filtering - skipping", path)
            continue

        vectorizer = TfidfVectorizer(stop_words='english')
        X_vectorized = vectorizer.fit_transform(df[q_col].astype(str))

        n_samples = X_vectorized.shape[0]
        knn_n = min(5, max(1, n_samples))
        knn = NearestNeighbors(n_neighbors=knn_n, metric='cosine')
        knn.fit(X_vectorized)

        models[topic] = {
            "df": df,
            "vectorizer": vectorizer,
            "knn": knn,
            "q_col": q_col,
            "display_cols": display_cols
        }
        logger.info("Trained retrieval model for topic '%s' (samples=%d, knn_n=%d)", topic, n_samples, knn_n)
    except Exception as e:
        logger.exception("Failed to process dataset %s: %s", path, e)
        continue

if not models:
    logger.error("FATAL: No retrieval models available. Please check CSV paths and columns.")
    raise SystemExit("No retrieval models available.")

# ----------------------
# Retrieval helpers
# ----------------------
def get_answer(question: str):
    """
    Returns: (best_answer_str, highest_similarity, best_match_data, best_topic)
    best_answer_str is assembled from display_cols.
    """
    best_answer_str = "I'm sorry, I couldn't find a relevant answer. Please try rephrasing your question."
    highest_similarity = -1.0
    best_topic = None
    best_match_data = None

    for topic, model in models.items():
        try:
            qv = model['vectorizer'].transform([question])
            distances, indices = model['knn'].kneighbors(qv, n_neighbors=1)
            similarity = 1 - float(distances[0][0])
        except Exception as e:
            logger.debug("Retrieval error for topic %s: %s", topic, e)
            continue

        if similarity > highest_similarity:
            highest_similarity = similarity
            best_topic = topic
            best_idx = int(indices[0][0])
            best_row = model['df'].iloc[best_idx]
            display_columns = model['display_cols']
            parts = []
            for col in display_columns:
                value = best_row.get(col, '')
                if pd.notna(value) and str(value).strip():
                    parts.append(str(value))
            best_answer_str = "\n\n".join(parts) if parts else best_answer_str
            best_match_data = best_row.to_dict()

    return best_answer_str, highest_similarity, best_match_data, best_topic


def get_additional_context(question: str, best_topic, best_match_data, max_related=2):
    """
    Return related texts (same topic + other topics) as a string.
    """
    if best_match_data is None:
        return ""
    context_parts = []

    if best_topic in models:
        model = models[best_topic]
        df = model['df']
        n_neigh = min(1 + max_related, len(df))
        try:
            qv = model['vectorizer'].transform([question])
            distances, indices = model['knn'].kneighbors(qv, n_neighbors=n_neigh)
            for i, idx in enumerate(indices[0][1:], 1):
                if i <= max_related:
                    related_row = df.iloc[int(idx)]
                    related_text = related_row.get(model['q_col'], '')
                    if pd.notna(related_text) and str(related_text).strip():
                        context_parts.append(f"Related ({best_topic}) {i}: {related_text}")
        except Exception as e:
            logger.debug("Error getting related items in same topic %s: %s", best_topic, e)

    for topic, model in models.items():
        if topic == best_topic:
            continue
        try:
            qv = model['vectorizer'].transform([question])
            distances, indices = model['knn'].kneighbors(qv, n_neighbors=1)
            similarity = 1 - float(distances[0][0])
            if similarity > 0.25:
                related_row = model['df'].iloc[int(indices[0][0])]
                related_text = related_row.get(model['q_col'], '')
                if pd.notna(related_text) and str(related_text).strip():
                    context_parts.append(f"Additional Context from {topic}: {related_text}")
        except Exception:
            continue

    return "\n".join(context_parts)

# ----------------------
# Gemini helpers / fallback
# ----------------------
def get_gemini_only_answer(question: str, allow_general=False, rag_level='medium'):
    """
    Fallback helper if gemini_model is unavailable.
    """
    if gemini_model is None:
        return ("Sorry — the server cannot generate answers on arbitrary topics right now because the language model is not available.")
    try:
        redirect_note = "Answer the user's question directly on the topic provided. Do not mention system limitations." if allow_general else "If unrelated, provide a short finance-related angle or ask for clarification."
        if rag_level == 'high':
            depth_note = "Provide an in-depth answer with reasoning and recommended actions."
        elif rag_level == 'low':
            depth_note = "Keep it very short (1-2 sentences)."
        else:
            depth_note = "Keep it concise (2-4 sentences)."

        prompt = (
            "You are FinAIgent.\n\n"
            f"User Question: {question}\n\n"
            f"{redirect_note}\n"
            f"{depth_note}\n\n"
            "Important output requirements:\n"
            "- Provide a complete answer. Do NOT truncate unless user asked for a summary.\n"
            "- Preserve paragraphs and line breaks.\n"
            "- Answer directly and only the user's question.\n"
            "- Use bullets / numbered steps when applicable.\n"
        )
        response = gemini_model.generate_content(prompt)
        if hasattr(response, 'text'):
            return response.text
        elif isinstance(response, dict):
            return response.get('output') or response.get('text') or str(response)
        else:
            return str(response)
    except Exception as e:
        logger.exception("Gemini-only answer failed: %s", e)
        return None

# ----------------------
# Domain / interaction helpers
# ----------------------
def is_finance_question(question: str, similarity_check_threshold: float = 0.25) -> bool:
    if not question or not question.strip():
        return False
    q = question.lower()
    keywords = {
        "bank", "card", "upi", "account", "pin", "otp", "phishing", "scam", "fraud",
        "loan", "investment", "credit", "debit", "transaction", "kyc", "password",
        "payment", "netbanking", "paytm", "phonepe", "google pay", "chargeback",
        "refund", "insurance", "broker", "ifsc", "imps", "neft", "rtgs", "savings", "savings account",
        "current account", "emi", "sip", "mutual fund", "fd", "fixed deposit", "tax", "insurance",
        # Added keywords for budgeting and personal finance
        "budget", "budgeting", "expense", "expenses", "save", "saving", "savings plan", 
        "emergency fund", "financial goal", "retirement", "pension", "ppf", "epf", "nps",
        "debt", "financial planning", "financial literacy", "money management",
        # Government schemes
        "pmjdy", "mudra", "apy", "pmjjby", "pmsby", "ayushman", "jan dhan", "kisan",
        "subsidy", "scheme", "yojana", "government scheme", "welfare",
    }
    for kw in keywords:
        if kw in q:
            return True
    
    # Allow translation-related queries if they contain language names or 'translate'
    translation_keywords = {'translate', 'marathi', 'hindi', 'tamil', 'telegu', 'bengali', 'gujarati', 'kannada', 'malayalam', 'punjabi'}
    if any(tk in q for tk in translation_keywords):
        return True

    try:
        for topic, model in models.items():
            qvec = model['vectorizer'].transform([question])
            dists, idxs = model['knn'].kneighbors(qvec, n_neighbors=1)
            sim = 1 - float(dists[0][0])
            if sim >= similarity_check_threshold:
                return True
    except Exception:
        return False
    return False


def is_greeting(question: str) -> bool:
    if not question:
        return False
    q = ''.join(ch for ch in question.strip().lower() if ch.isalnum() or ch.isspace())
    tokens = q.split()
    greetings = {'hi','hello','hey','hiya','goodmorning','goodafternoon','goodevening','gm'}
    if 1 <= len(tokens) <= 3 and any(t in greetings for t in tokens):
        return True
    if tokens and tokens[0] in greetings:
        return True
    return False


def is_small_talk(question: str):
    if not question:
        return False
    q = ''.join(ch for ch in question.strip().lower() if ch.isalnum() or ch.isspace())
    tokens = q.split()
    if any(t in tokens for t in ['thanks','thank','thx','ty']):
        return 'thanks'
    if any(t in tokens for t in ['bye','goodbye','see you','cya','later']):
        return 'farewell'
    if any(t in tokens for t in ['ok','okay','cool','great','nice']):
        return 'ack'
    return False


def auto_select_rag(question: str, is_finance=False) -> str:
    if not question or not question.strip():
        return 'medium'
    q = question.lower().strip()
    analytic_tokens = [
        'analyze','analysis','explain','explanation','evaluate','compare',
        'contrast','detailed','in-depth','how to','steps','strategy',
        'recommend','risk','risks','implications','pros and cons','advantages',
        'disadvantages','why','cause','impact','breakdown','budget','plan','calculate'
    ]
    quick_tokens = [
        'what is','define','meaning of','when','which bank','ifsc','how much','how many','balance','limit'
    ]
    words = q.split()
    wc = len(words)
    for t in analytic_tokens:
        if t in q:
            return 'high'
    if wc <= 4:
        if is_finance and any(x in q for x in ['advice','should i','invest','loan']):
            return 'medium'
        return 'low'
    if 5 <= wc <= 15:
        for t in quick_tokens:
            if t in q:
                return 'low'
        return 'medium'
    if wc > 15:
        return 'high'
    return 'medium'

# ----------------------
# Routes
# ----------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json(silent=True) or {}
    raw_question = data.get('question', '') or request.form.get('question', '') or request.args.get('question', '')
    forced_lang = (data.get('lang') or request.form.get('lang') or request.args.get('lang') or '').strip()
    question_orig = (raw_question or '').strip()

    if not question_orig:
        return jsonify({'answer': "Please provide a question in the request."})

    # Detect user language (or use forced)
    user_lang = forced_lang.lower() if forced_lang else detect_language_of_text(question_orig, default='en')
    logger.info("Detected user language: %s", user_lang)
    need_translation = user_lang != 'en'
    question_for_processing = question_orig
    detected_src_lang = user_lang

    # Translate user question to English for retrieval/LLM if needed
    if need_translation:
        try:
            translated_q, detected_src = translate_text(question_orig, src='auto', dest='en')
            if translated_q and translated_q.strip():
                question_for_processing = translated_q
                detected_src_lang = detected_src or user_lang
                logger.info("Translated incoming question from %s -> en", detected_src_lang)
            else:
                logger.info("Translation returned empty; using original for processing.")
        except Exception as e:
            logger.warning("Translation failed; using original question. Error: %s", e)
            question_for_processing = question_orig

    # Use question_for_processing (English) for retrieval and LLM prompt
    question = question_for_processing.strip()

    # Greeting / small-talk
    if is_greeting(question):
        reply = "Hi! How can I help?"
        if need_translation:
            try:
                r_t, _ = translate_text(reply, src='en', dest=detected_src_lang)
                return jsonify({'answer': r_t})
            except Exception:
                return jsonify({'answer': reply})
        return jsonify({'answer': reply})

    small = is_small_talk(question)
    if small:
        if small == 'thanks':
            resp = "You're welcome!"
        elif small == 'farewell':
            resp = "Goodbye!"
        else:
            resp = "Okay."
        if need_translation:
            try:
                r_t, _ = translate_text(resp, src='en', dest=detected_src_lang)
                return jsonify({'answer': r_t})
            except Exception:
                return jsonify({'answer': resp})
        return jsonify({'answer': resp})

    # Block non-finance questions (with translation fallback)
    is_finance = is_finance_question(question)
    
    # If the question is in a foreign language and translation failed, 
    # the finance check might fail. We'll be more lenient if need_translation is True.
    if not is_finance and not need_translation:
        refusal_en = "Sorry — I can only answer finance-related questions. Please ask a finance or payment-related question."
        if need_translation:
            try:
                refusal_t, _ = translate_text(refusal_en, src='en', dest=detected_src_lang)
                return jsonify({'answer': refusal_t})
            except Exception:
                return jsonify({'answer': refusal_en})
        return jsonify({'answer': refusal_en})

    # Finance flow: RAG + Gemini — with dataset-translation-aware behavior
    rag_level = auto_select_rag(question, is_finance=True)
    logger.info("RAG Level: %s, Similarity Check...", rag_level)

    base_answer, similarity_score, best_match_data, best_topic = get_answer(question)
    logger.info("Best topic: %s, Similarity: %.4f", best_topic, similarity_score)
    
    # DECISION LOGIC: Should we use RAG context or let LLM use its own knowledge?
    # If similarity is very high (>0.6), definitely use RAG.
    # If similarity is medium (0.3-0.6), use RAG + Gemini context.
    # If similarity is low (<0.3), let Gemini answer based on its general knowledge but grounded in finance.
    
    context_to_use = ""
    if similarity_score > 0.25:
        context_to_use = base_answer
        additional_context = get_additional_context(question, best_topic, best_match_data, max_related=(3 if rag_level == 'high' else 1))
        if additional_context:
            context_to_use += "\n\nAdditional Related Info:\n" + additional_context
    else:
        logger.info("Similarity too low (%.4f). Relying on Gemini general knowledge.", similarity_score)

    # Build prompt:
    prompt_parts = [
        "You are FinAIgent, a professional financial assistant.",
        "Your goal is to provide accurate, helpful, and concise financial advice.",
        "",
        "User Question:",
        question,
        ""
    ]

    if context_to_use:
        prompt_parts += ["Supportive Knowledge from our Database:", context_to_use, ""]
    else:
        prompt_parts += ["Note: Our specific database doesn't have a direct match, so please answer using your general professional financial knowledge.", ""]

    # Add strict output requirements
    output_requirements = [
        "Guidelines:",
        "- If context was provided, prioritize its facts but explain them clearly.",
        "- If context was NOT provided, answer accurately using your internal training data on finance.",
        "- Provide a professional, informative, and direct answer.",
        "- Do NOT mention 'retrieved documents' or 'the context'. Just answer the user directly.",
    ]
    prompt_parts += output_requirements

    # If user language is not English, instruct Gemini to respond in that language
    if need_translation and user_lang:
        prompt_parts += ["", f"Respond in this language: {user_lang}", f"(Do not add translations into other languages; respond only in {user_lang})."]

    prompt = "\n".join(part for part in prompt_parts if part is not None)

    final_answer = None

    # Primary: call Gemini with the constructed prompt
    if gemini_model is not None:
        try:
            response = gemini_model.generate_content(prompt)
            try:
                final_answer = response.text
            except (AttributeError, ValueError):
                final_answer = None
        except Exception as e:
            logger.exception("Gemini call failed: %s", e)
            final_answer = None

    # Fallback to helper if Gemini missing or failed
    if not final_answer:
        # If similarity was low, we allow general answering if Gemini is available
        allow_general_flag = (similarity_score < 0.3)
        final_answer = get_gemini_only_answer(question, allow_general=allow_general_flag, rag_level=rag_level)
        
    # Final safety: If both failed, use the base retrieval answer
    if not final_answer or "Sorry — I couldn't generate" in str(final_answer):
        if base_answer and similarity_score > 0.2:
            final_answer = base_answer
        else:
            # If we still have nothing, try one last very simple Gemini call
            if gemini_model:
                try:
                    simple_resp = gemini_model.generate_content(f"Answer briefly in {user_lang}: {question_orig}")
                    final_answer = simple_resp.text
                except:
                    final_answer = "Sorry — I couldn't generate an answer right now. Please try again later."
            else:
                final_answer = "Sorry — I couldn't generate an answer right now. Please try again later."

    # If Gemini answered in English but user needed non-English and pre-translation was not done,
    # fall back to translating final answer into user's language
    # (Prefer not to double-translate if Gemini already responded in user's language.)
    if need_translation and final_answer:
        # detect language of final_answer; if different from user_lang, translate; else keep as-is
        final_lang = None
        try:
            final_lang = detect_language_of_text(final_answer, default='en') if detect_lang else None
        except Exception:
            final_lang = None

        if final_lang and final_lang.lower().split('-')[0] == user_lang.lower().split('-')[0]:
            # already in user's language
            return jsonify({'answer': final_answer})
        else:
            try:
                final_translated, _ = translate_text(final_answer, src='auto', dest=detected_src_lang)
                if final_translated and final_translated.strip():
                    return jsonify({'answer': final_translated})
                else:
                    return jsonify({'answer': final_answer})
            except Exception as e:
                logger.warning("Final answer translation failed: %s", e)
                return jsonify({'answer': final_answer})
    
    if not final_answer:
        final_answer = "Sorry, I am unable to process that request right now."

    # If no translation needed, return final answer as-is
    return jsonify({'answer': final_answer})

# ----------------------
# Run server
# ----------------------
if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    logger.info("Starting FinAIgent on port %d", port)
    app.run(host='0.0.0.0', port=port, debug=True)