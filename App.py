# -*- coding: utf-8 -*-

import streamlit as st
from gtts import gTTS
from io import BytesIO
from PIL import Image
from googletrans import Translator
from transformers import BlipProcessor, BlipForConditionalGeneration


def main():
    
    # Configuración de la página
    st.set_page_config(
        page_title="Clothespeaker",
        page_icon="👗",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    

    def image_to_text(image):
        """Función que convierte de imagen a texto"""
        
        # Cargar modelo de Hugging Face
        modelName = "Luna288/image-captioning-clothes"
        processor = BlipProcessor.from_pretrained(modelName)
        model = BlipForConditionalGeneration.from_pretrained(modelName)
    
        # Preprocesar imagen
        inputs = processor(image, return_tensors="pt")
        generated_ids = model.generate(pixel_values=inputs.pixel_values, max_length=150)
    
        # Generar texto a partir de la imagen
        caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return caption

   
    # Generación de Audio
    def text_to_speech(text):
        """"Función que convierte texto a audio"""
        tts = gTTS(text, lang="es")
        
        # Guardar audio en BytesIO
        audio_file = BytesIO()
        tts.write_to_fp(audio_file)
        audio_file.seek(0)
        return audio_file
    
    # Texto de las instrucciones
    instrucciones = """
    Bienvenido al conversor de imagen de ropa a audio. 
    Primero, sube una imagen desde tu dispositivo. 
    Luego de un minuto, se generará una descripción de la imagen que podrás escuchar y ajustar su velocidad.
    """
    
    # Título de la aplicación
    st.title("Conversor de imagen de ropa a audio")
    st.markdown(
        """
        ##### Esta aplicación permite a los usuarios subir una imagen de ropa y convertirla a audio, es posible ajustar la velocidad del mismo.
        """
    )
    
    # Botón de audio para las instrucciones
    st.markdown("### Instrucciones de uso")
    instructions_audio = text_to_speech(instrucciones)
    st.audio(instructions_audio, format='audio/mp3')
    
    # Subir imagen
    st.markdown("#### Sube una imagen por favor 😊")
    uploaded_image = st.file_uploader('Imagen de ropa:',type=["jpg", "jpeg", "png"])
    
    
    # Procesar la imagen subida
    if uploaded_image is not None:
        # Mostrar la imagen cargada
        image = Image.open(uploaded_image)
        st.image(image, caption='Imagen Subida', use_column_width=False, width=300)
    
        # Generar texto a partir de la imagen
        translator = Translator()
        with st.spinner("Procesando imagen para generar texto..."):
            generated_text = image_to_text(image)
            generated_text = translator.translate(generated_text, dest='es').text
            st.success("¡Texto generado!")
            st.markdown(f"**Texto generado:** {generated_text}")
    
        # Generar audio a partir del texto
        with st.spinner("Generando audio..."):
            audio_file = text_to_speech(generated_text)
            st.audio(audio_file, format='audio/mp3')
            
    st.markdown("Clothespeaker puede contener errores.")

if __name__ == "__main__":
    main()
