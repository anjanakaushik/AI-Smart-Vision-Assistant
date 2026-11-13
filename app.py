import streamlit as st
from ultralytics import YOLO
from PIL import Image
from gtts import gTTS
import cv2
import speech_recognition as sr
import io

st.set_page_config(
    page_title="Blind Assistance System",
    page_icon="👁️"
)

st.title("AI-Based Blind Assistance System")
st.subheader("Object Detection for Visually Impaired People")

st.write("Detect objects using AI and receive voice assistance.")

language = st.selectbox(
    "Choose Your Language",
    ["English", "Hindi"]
)

st.header("Emergency Assistance 🚨")

if st.button("🚨 Emergency Alert"):
    st.error("Emergency! Please seek help.")

    emergency_message = (
        "मुझे मदद की जरूरत है। कृपया मेरी सहायता करें।"
        if language == "Hindi"
        else "I need help. Please assist me."
    )

    emergency_speech = gTTS(
        text=emergency_message,
        lang="hi" if language == "Hindi" else "en"
    )

    emergency_speech.save("emergency_alert.mp3")

    st.audio("emergency_alert.mp3", format="audio/mp3")
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

st.header("Choose Image Input")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

camera_image = st.camera_input("Capture a photo")

image_file = camera_image if camera_image is not None else uploaded_file

st.header("Voice Commands 🎤")

audio_value = st.audio_input("Speak your command")

command = ""

if audio_value is not None:
    recognizer = sr.Recognizer()

    try:
        audio_file = sr.AudioFile(
            io.BytesIO(audio_value.getvalue())
        )

        with audio_file as source:
            audio_data = recognizer.record(source)

        speech_language = "hi-IN" if language == "Hindi" else "en-IN"

        command = recognizer.recognize_google(
            audio_data,
            language=speech_language
        ).lower()

        st.success("You said: " + command)

    except sr.UnknownValueError:
        st.warning("Sorry, I could not understand your voice.")

    except sr.RequestError:
        st.error("Internet connection required.")

detect_button = st.button("Detect Objects")

voice_detect = any(
    word in command
    for word in [
        "detect",
        "identify",
        "what can you see",
        "पहचानो",
        "बताओ",
        "क्या दिख रहा है"
    ]
)

if detect_button or voice_detect:

    if image_file is None:
        st.warning("Please upload an image or capture a photo.")

    else:
        image = Image.open(image_file).convert("RGB")

        results = model(image)

        object_counts = {}
        detected_objects = []

        for box in results[0].boxes:

            class_id = int(box.cls[0])
            name = model.names[class_id]
            confidence = float(box.conf[0]) * 100

            object_counts[name] = object_counts.get(name, 0) + 1

            coordinates = box.xyxy[0].tolist()
            object_height = coordinates[3] - coordinates[1]
            coordinates = box.xyxy[0].tolist()

            x_center = (coordinates[0] + coordinates[2]) / 2

            if x_center < image.width / 3:
               position = "Left"
            elif x_center < 2 * image.width / 3:
               position = "Centre"
            else:
               position = "Right"

            object_height = coordinates[3] - coordinates[1]

            size_ratio = object_height / image.height

            if size_ratio > 0.50:
                proximity = "Near"
            elif size_ratio > 0.25:
                proximity = "Medium"
            else:
                proximity = "Far"

            detected_objects.append(
                (name, confidence, proximity,position)
            )

        result_image = results[0].plot()
        result_image = cv2.cvtColor(
            result_image, cv2.COLOR_BGR2RGB
        )

        st.image(result_image, caption="Detection Results")

        if detected_objects:

            st.success("Objects detected!")

            st.header("Object Count")

            for name, count in object_counts.items():
                st.write(f"{name}: {count}")

            st.header("Confidence and Proximity")

            for name, confidence, proximity, position in detected_objects:
                st.write(
                    f"{name} | Confidence: {confidence:.2f}% | "
                    f"Proximity: {proximity} | Position: {position}"
                )

            if language == "Hindi":

                hindi_names = {
                    "person": "व्यक्ति",
                    "chair": "कुर्सी",
                    "bottle": "बोतल",
                    "car": "कार",
                    "bus": "बस",
                    "dog": "कुत्ता",
                    "cat": "बिल्ली",
                    "cell phone": "मोबाइल फोन",
                    "laptop": "लैपटॉप",
                    "book": "किताब",
                    "cup": "कप",
                    "dining table": "खाने की मेज",
                    "potted plant": "गमले का पौधा"
                }

                hindi_objects = []

                for name, count in object_counts.items():
                    hindi_name = hindi_names.get(name, name)
                    hindi_objects.append(f"{count} {hindi_name}")

                message = (
                    "मुझे दिखाई दे रहा है "
                    + " और ".join(hindi_objects)
                )

                voice_language = "hi"

            else:

                message = "I can see " + ", ".join(
                    f"{count} {name}"
                    for name, count in object_counts.items()
                )

                voice_language = "en"

            if language == "Hindi":
                position_names = {
                    "Left": "बाईं ओर",
                    "Centre": "बीच में",
                    "Right": "दाईं ओर"
                }

                position_message = "। ".join(
                    f"{name} {position_names[position]}"
                    for name, confidence, proximity, position in detected_objects
                )

                message += "। वस्तुओं की स्थिति: " + position_message

            else:
                position_message = ", ".join(
                    f"{name} is on the {position.lower()}"
                    for name, confidence, proximity, position in detected_objects
                )

                message += ". Object positions: " + position_message

            st.write("Voice Message:", message)

            speech = gTTS(
                text=message,
                lang=voice_language
            )

            speech.save("voice_output.mp3")

            st.subheader("Voice Assistance 🔊")

            st.audio(
                "voice_output.mp3",
                format="audio/mp3",
                autoplay=True
            )

        else:
            st.warning("No objects detected.")
# AI SCENE DESCRIPTION

@st.cache_resource
def load_caption_model():
    from transformers import BlipProcessor, BlipForConditionalGeneration

    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    caption_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    return processor, caption_model


st.header("AI Scene Description")

if st.button("Describe My Surroundings"):

    if image_file is None:
        st.warning("Please upload an image or capture a photo.")

    else:
        with st.spinner("Understanding the scene..."):

            processor, caption_model = load_caption_model()

            scene_image = Image.open(
                io.BytesIO(image_file.getvalue())
            ).convert("RGB")

            inputs = processor(
                images=scene_image,
                return_tensors="pt"
            )

            output = caption_model.generate(
                **inputs,
                max_new_tokens=50
            )

            description = processor.decode(
                output[0],
                skip_special_tokens=True
            )

            st.success("Scene Description:")
            st.write(description)

            scene_speech = gTTS(
                text=description,
                lang="en"
            )

            scene_speech.save("scene_description.mp3")

            st.audio(
                "scene_description.mp3",
                format="audio/mp3",
                autoplay=True
            )
