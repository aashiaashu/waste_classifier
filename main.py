from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import streamlit as st
from dotenv import load_dotenv

def classify_waste(img):
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    model = load_model("keras_model.h5", compile=False)

    # Load the labels
    class_names = open("labels.txt", "r").readlines()

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = img.convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    # print("Class:", class_name[2:], end="")
    # print("Confidence Score:", confidence_score)

    return class_name, confidence_score

st.set_page_config(layout='wide')

st.title("Waste Classifier Sustainability App")

input_img = st.file_uploader("Enter your image", type=['jpg', 'png', 'jpeg'])

if input_img is not None:
    if st.button("Classify"):

        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            st.info("Your uploaded Image")
            st.image(input_img, use_column_width=True)

        with col2:
            st.info("Your Result")
            image_file = Image.open(input_img)
            label, confidence_score = classify_waste(image_file)
            col4, col5 = st.columns([1, 1])
            if label == "0 cardboard\n":
                st.success("The image is classified as CARDBOARD.")
                with col4:
                    st.image("sdg goals/12.png", use_column_width=True)
                    st.image("sdg goals/13.png", use_column_width=True)
                with col5:
                    st.image("sdg goals/14.png", use_column_width=True)
                    st.image("sdg goals/15.png", use_column_width=True)
            elif label == "1 plastic\n":
                st.success("The image is classified as PLASTIC.")
                with col4:
                    st.image("sdg goals/6.jpg", use_column_width=True)
                    st.image("sdg goals/12.png", use_column_width=True)
                with col5:
                    st.image("sdg goals/14.png", use_column_width=True)
                    st.image("sdg goals/15.png", use_column_width=True)
            elif label == "2 glass\n":
                st.success("The image is classified as GLASS.")
                with col4:
                    st.image("sdg goals/12.png", use_column_width=True)
                with col5:
                    st.image("sdg goals/14.png", use_column_width=True)
            elif label == "3 metal\n":
                st.success("The image is classified as METAL.")
                with col4:
                    st.image("sdg goals/3.png", use_column_width=True)
                    st.image("sdg goals/6.jpg", use_column_width=True)
                with col5:
                    st.image("sdg goals/12.png", use_column_width=True)
                    st.image("sdg goals/14.png", use_column_width=True)
            else:
                st.error("The image is not classified as any relevant class.")

        with col3:
            st.info("Environmental Impact Information")
            if label == "0 cardboard\n":
                st.write(
                    "Cardboard production involves cutting down trees, contributing to deforestation and habitat loss. However, cardboard is renewable, and responsible forestry practices can mitigate environmental impact. Cardboard is also highly recyclable, providing an opportunity to reduce the demand for new materials. Emphasizing recycling programs, sustainable sourcing of raw materials, and reducing waste through efficient packaging designs are key strategies to enhance the environmental profile of cardboard.")
            elif label == "1 plastic\n":
                st.write(
                    "Plastic production is associated with environmental concerns due to its reliance on fossil fuels and non-biodegradable nature. The extraction of petroleum for plastic production contributes to habitat destruction, and plastic waste poses a significant threat to ecosystems. Recycling plastic is essential to reduce its environmental impact, but challenges such as contamination and low recycling rates persist. Sustainable alternatives, improved waste management, and efforts to reduce single-use plastics are crucial for addressing the environmental challenges posed by plastic.")
            elif label == "2 glass\n":
                st.write(
                    "Glass has a dual environmental impact. Its production, involving raw material extraction and energy-intensive processes, contributes to emissions and habitat disruption. On the positive side, glass is highly recyclable and durable, reducing the need for new production. To maximize its eco-friendliness, emphasis should be placed on recycling efforts, sustainable manufacturing practices, and minimizing transportation-related impacts by sourcing locally.")
            elif label == "3 metal\n":
                st.write(
                    "The environmental impact of metal production involves resource extraction, energy-intensive processes, and potential emissions. Mining and refining metals can lead to habitat disruption and contribute to greenhouse gas emissions. However, metals are highly recyclable, and recycling can significantly reduce the need for new production, saving energy and mitigating environmental impact. Sustainable practices in mining, coupled with increased recycling efforts, are crucial for minimizing the overall environmental footprint of metal.")
            else:
                st.write("No environmental impact information available for this class.")

