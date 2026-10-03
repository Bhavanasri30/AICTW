import streamlit as st

st.title("Media Upload Application")
st.sidebar.header("Upload Options")
option = st.sidebar.selectbox(
    "Choose File Type",
    ["Image","Audio","Video"]
)
uploaded_file = st.file_uploader(
    "Upload File",
    type=["jpg","jpeg","png","mp3","wav","mp4"]
)
camera_photo = st.camera_input(
    "Take a Picture"
)
if uploaded_file is not None:
    if option=="Image":
        st.image(
            uploaded_file,
            caption="Uploaded Image",
            use_container_width=True
        )
    elif option=="Audio":
        st.audio(uploaded_file)
    elif option=="Video":
        st.video(uploaded_file)
if camera_photo is not None:
    st.image(
        camera_photo,
        caption="Captured Image",
        use_container_width=True
    )
st.write("Upload complete.")