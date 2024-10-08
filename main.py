import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.resnet_v2 import preprocess_input, decode_predictions


#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("RESNET152V2_plant_disease (6).h5")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(224,224))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr=np.expand_dims(input_arr,axis=0)
    input_arr=input_arr/255
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

st.markdown("## PLANT DISEASE RECOGNITION SYSTEM:herb:")
tab1, tab2, tab3 = st.tabs(["Home", "About", "Disease Recognition"])

    #Main Page
with tab1:
    image_path = "https://www.ischool.berkeley.edu/sites/default/files/styles/fullscreen/public/sproject_teaser_image/reversed.jpg?itok=ShdqgpJr"  
    st.image(image_path,use_column_width=True)
    st.markdown("""
        Welcome to the Plant Leaf Disease Recognition System! 
        The analysis of images for plant disease diagnosis is a rapidly developing area with significant potential. This application provides a user-friendly platform to leverage machine learning algorithms for the identification of potential plant leaf diseases. By uploading an image of the affected area, users can receive a preliminary diagnosis, aiding in informed decision-making regarding plant health management strategies.

        ### How It Works
        1.	**Upload Image:** On the Disease Recognition tab, the user can upload any image less than 200MB in size of the affected area, through a simple drag and drop or by browsing through their files. Following the upload, click on the predict button.
        2.	**Analysis:** This app uses RESNET model to analyse photos, identifying potential plant diseases based on visual patterns like Colour Changes, Lesions and Spots, Discolouration.
        3.	**Results:** The system provides the most probable diagnosis and offers a link to a Google search for that specific disease, allowing you to verify the information.

        ### Get Started
        Click on the **Disease Recognition** tab to upload an image and experience the Plant Leaf Disease Recognition System!
        """)

#About Project
with tab2:
    st.markdown("### About Dataset")
    image_path="about_image.png"
    st.image(image_path,use_column_width=True)
    st.markdown("""
                This dataset can be found on https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset. 
                
                This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purpose.
                #### Content
                1. train (70295 images)
                2. test (33 images)
                3. validation (17572 images)
                """)

#Prediction Page
with tab3:
    st.header("Disease Recognition:mag:")
    test_image = st.file_uploader(
        "Choose an Image:",
        key="file",
        accept_multiple_files=False,
        type=["jpg", "jpeg", "png"],
    )
    if test_image is not None:
        st.image(test_image, width=300)
    
    # Predict button
    if test_image is not None:
        if(st.button("Predict")):
            max_upload_size = 200 * 1024 * 1024 # 200MB in bytes
            if test_image.size > max_upload_size:
                st.error(f"File size exceeds the maximum allowed (200MB). Please choose a smaller file.")
            else:
                # Use st.spinner to display a loader while making predictions
                with st.spinner("Predicting..."):
                    result_index = model_prediction(test_image)
                    class_names = ['Apple Scab','Apple Black Rot',
                                    'Apple Cedar Rust','Apple healthy',
                                    'Blueberry healthy','Cherry (including sour) Powdery Mildew',
                                    'Cherry (including sour) healthy','Corn (maize) Cercospora Leaf Spot Gray Leaf Spot',
                                    'Corn (maize) Common Rust','Corn (maize) Northern Leaf Blight',
                                    'Corn (maize) healthy','Grape Black Rot','Grape Esca (Black Measles)',
                                    'Grape Leaf Blight (Isariopsis Leaf Spot)','Grape healthy',
                                    'Orange Huanglongbing (Citrus Greening)','Peach Bacterial Spot',
                                    'Peach healthy','Pepper bell Bacterial Spot',
                                    'Pepper bell healthy','Potato Early Blight',
                                    'Potato Late Blight','Potato healthy',
                                    'Raspberry healthy','Soybean healthy',
                                    'Squash Powdery Mildew','Strawberry Leaf Scorch',
                                    'Strawberry healthy','Tomato Bacterial Spot',
                                    'Tomato Early Blight','Tomato Late Blight',
                                    'Tomato Leaf Mold','Tomato Septoria Leaf Spot',
                                    'Tomato Spider Mites (Two-Spotted Spider Mite)','Tomato Target Spot',
                                    'Tomato Yellow Leaf Curl Virus','Tomato Mosaic Virus',
                                    'Tomato healthy']

                st.write("Our Prediction")
                st.success("Model is Predicting it's a {}".format(class_names[result_index]))
                st.balloons()
                # Add Google Search button for the top prediction
                google_search_url = f"https://www.google.com/search?q={class_names[result_index].replace(' ', '+')}"
                google_icon_url = "https://clipartcraft.com/images/transparent-background-google-logo-6.png"
                google_button_html = f'<a href="{google_search_url}" target="_blank" style="text-decoration: none;"><img src="{google_icon_url}" alt="Google Icon" width="20" height="20" style="margin-right: 5px;">Search on Google for {class_names[result_index]} </a>'
                st.markdown(google_button_html, unsafe_allow_html=True)