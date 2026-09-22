# import path
from pathlib import Path

# application settings
APP_NAME = "Nurture Space"
APP_VERSION = "0.1.1"
APP_ICON = "🌸"
# project supports english interaction.
DEFAULT_LANGUAGE = "English"
DEFAULT_LANGUAGE_CODE = "en"
# enables additional debugging during development.
DEVELOPMENT_MODE = False

#audio settings
# voice and video reflections are limited to 60 seconds.
MAX_RECORDING_SECONDS = 60

# audio is processed as 16000 herts and  mono for speech to text
# and audio emotion analysis.
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1

# video related settings
# 5 representative frames are sampled from each video
# for the visible expression ( video ) analysis.
VIDEO_EXPRESSION_SAMPLE_FRAMES = 5

# 3 valid frame level face predictions are required at least
#before visible expression evidence is used.
MIN_VIDEO_EXPRESSION_FACES = 3

# privacy settings
# version of the privacy information accepted by the user.
CURRENT_PRIVACY_VERSION = "1.0"

# project path
BASE_DIR = Path(__file__).resolve().parent.parent

TEMP_DATA_DIR = BASE_DIR /"data"/"temp"
DATABASE_DIR = BASE_DIR /"data"/"database"
DATABASE_PATH = DATABASE_DIR /"postpartum_wellbeing.db"
THEME_RULES_PATH = BASE_DIR /"config"/"theme_rules.json"
SAFETY_RESOURCES_PATH = BASE_DIR /"config"/"safety_resources.json"
CSS_PATH = BASE_DIR /"flask_app"/"static"/"css"/"app.css"

# selected AI models
# final pretrained models selected for each processing component.
SELECTED_TEXT_EMOTION_MODEL = "joeddav/distilbert-base-uncased-go-emotions-student"
SELECTED_AUDIO_EMOTION_MODEL = "Khoa/w2v-speech-emotion-recognition"
SELECTED_SPEECH_TO_TEXT_MODEL = "small.en"
SELECTED_VIDEO_EXPRESSION_MODEL = "enet_b0_8_va_mtl"
# local language model used by the bounded AI assistant and dashboard text generation functions.
SELECTED_LLM_MODEL = "qwen3:4b-instruct"
# Sentence embedding model used for faiss based
# approved resource retrieval.
SELECTED_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


# reflection modes and application features enabled
ENABLE_TEXT_REFLECTION = True
ENABLE_VOICE_REFLECTION = True
ENABLE_VIDEO_REFLECTION = True
ENABLE_AI_ASSISTANT = True
ENABLE_USER_ACCOUNTS = True

# questionnaire settings
# five point scale.
CHECKIN_MIN_VALUE = 1
CHECKIN_MAX_VALUE = 5

# 4 core responses are needed to calculate the question score and show indicator.
REQUIRED_CHECKIN_ANSWERS = 4
# threshold applied to the mean 4 questions score.
# stress is reversed before the mean is calculated.
STEADY_STATUS_MINIMUM = 3.67
SOME_STRAIN_STATUS_MINIMUM = 2.34