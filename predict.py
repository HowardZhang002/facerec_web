import time
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from retinaface import Retinaface
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode
from streamlit_webrtc import webrtc_streamer
import threading

st.set_page_config(
    page_title = "人脸识别",
    page_icon = ":camera:",
    layout = "centered",
)

session_state = st.session_state
st.title("基于深度学习的人脸识别平台")
st.markdown("""
<style>
.css-9s5bis.edgvbvh3
{
    visibility:hidden;
}
.css-h5rgaw.egzxvld1
{
    visibility:hidden;
}
.css-yyj0jg.edgvbvh3
{
    visibility:hidden;
}
.css-14xtw13.e8zbici0
{
    visibility:hidden;
}
</style>
""",unsafe_allow_html=True)
# st.write(session_state)


image_captured = st.camera_input("视频输入",key="firstCamera")
retinaface = Retinaface()
if image_captured is not None:
    st.session_state['image'] = image_captured
    col1, col2 = st.columns(2)
    with col1:
        st.image(caption = "输入图片",image = image_captured)
    image = Image.open(image_captured)
    image = np.array(image)
    r_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    r_image = retinaface.detect_image(image)
    # r_image = cv2.cvtColor(r_image, cv2.COLOR_RGB2BGR)
    with col2:
        st.image(caption = "人脸识别",image = r_image)


from streamlit_webrtc import VideoTransformerBase, webrtc_streamer


class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = np.array(retinaface.detect_image(img))
        return img

st.subheader("实时监控")
webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

if __name__ == "__main__":
    retinaface = Retinaface()
    mode = "predict"
    video_path      = 0
    video_save_path = ""
    video_fps       = 25.0

    if mode == "predict":
        while True:
            img = input('Input image filename:')
            image = cv2.imread(img)
            if image is None:
                print('Open Error! Try again!')
                continue
            else:
                image   = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                r_image = retinaface.detect_image(image)
                r_image = cv2.cvtColor(r_image,cv2.COLOR_RGB2BGR)
                cv2.imshow("after",r_image)
                cv2.waitKey(0)

    elif mode == "video":
        capture = cv2.VideoCapture(video_path)
        if video_save_path!="":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        fps = 0.0
        while(True):
            t1 = time.time()
            ref,frame=capture.read()
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            frame = np.array(retinaface.detect_image(frame))
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
                    
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("video",frame)
            c= cv2.waitKey(1) & 0xff 
            if video_save_path!="":
                out.write(frame)

            if c==27:
                capture.release()
                break
        capture.release()
        out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        test_interval = 100
        img = cv2.imread('img/obama.jpg')
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        tact_time = retinaface.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video' or 'fps'.")
