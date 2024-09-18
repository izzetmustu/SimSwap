
import cv2
import torch
import numpy as np
from models.models import create_model
from options.recognize_options import RecognizeOptions
from insightface_func.face_detect_crop_multi import Face_detect_crop
import os
from util.norm import SpecificNorm
from parsing_model.model import BiSeNet
import json
from util import load_database, get_latent, find_closest, swap_faces, update_face_history

def main(db_not_replaced:dict, db_to_replace:dict):
  opt = RecognizeOptions().parse()
  opt.no_simswaplogo = True
  
  if opt.use_mask:
    n_classes = 19
    mask_net = BiSeNet(n_classes=n_classes)
    # mask_net.cuda()
    save_pth = os.path.join('./parsing_model/checkpoint', '79999_iter.pth')
    mask_net.load_state_dict(torch.load(save_pth))
    mask_net.cuda()
    mask_net.eval()
  else:
    mask_net =None  
  
  torch.nn.Module.dump_patches = True
  mode = 'None'
  model = create_model(opt)
  model.eval()
  spNorm = SpecificNorm()
  app = Face_detect_crop(name='antelope', root='./insightface_func/models')
  app.prepare(ctx_id= 0, det_thresh=0.6, det_size=(224,224),mode=mode)

  threshold = 0.3
  
  # Create pipeline
  pipeline = dai.Pipeline()

  # Define source and outputs
  camRgb = pipeline.create(dai.node.ColorCamera)
  xoutVideo = pipeline.create(dai.node.XLinkOut)
  xoutPreview = pipeline.create(dai.node.XLinkOut)

  xoutVideo.setStreamName("video")
  xoutPreview.setStreamName("preview")

  # Properties
  camRgb.setPreviewSize(300, 300)
  camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
  camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
  camRgb.setInterleaved(True)
  camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

  # Linking
  camRgb.video.link(xoutVideo.input)
  camRgb.preview.link(xoutPreview.input)

  # Connect to device and start pipeline
  with dai.Device(pipeline) as device:

    video = device.getOutputQueue('video')
    preview = device.getOutputQueue('preview')  

    videoFrame = video.get()
    previewFrame = preview.get()
    
    # Get the frame width, height, and frames per second (fps)
    width = int(videoFrame.getCvFrame().shape[1])
    height = int(videoFrame.getCvFrame().shape[0])
    # print video properties
    
    # VideoWriter to save processed video frames
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 25
    out = cv2.VideoWriter(os.path.join(opt.output_path, 'luxonis.mp4'), fourcc, fps, (width, height))
    
    # Initialize a history dictionary to store the last used embedding for unknown faces
    face_history = {}

    while true:
      videoFrame = video.get()
      previewFrame = preview.get()
      frame = videoFrame.getCvFrame()

      with torch.no_grad():
        results = app.get_custom(frame, opt.crop_size)
        if results is not None:
          faces, mats, bboxes = results
          faces_to_replace = []
          mats_to_replace = []
          closest_unknowns = []

          for face, mat, bbox in zip(faces, mats, bboxes):
            face_embedding = get_latent(face, model)

            # Find the best match in the database
            best_match_name, closest_latent, best_match_score, best_match_index = find_closest(face_embedding, db_not_replaced)

            # Recognize known face (if above threshold)
            if best_match_score > threshold:
              # print(f"Recognized {best_match_name} with confidence: {best_match_score}")
              cv2.putText(frame, f"{best_match_name}", (int(bbox[0]), int(bbox[1] - 10)),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
              # Unknown face detected, apply face swap
              # print("Unknown person detected. Swapping face.")
              _, closest_unknown, _, _ = find_closest(face_embedding, db_to_replace)
              closest_unknown = torch.Tensor(closest_unknown).cuda()
              # TO DO
              # CHECK IF THE PERSON IS THE SAME AS THE ONE IN THE PREVIOUS FRAME          
              consistent_latent = update_face_history(face_history,face_embedding, closest_unknown, bbox)
      
              faces_to_replace.append(face)
              mats_to_replace.append(mat)
              closest_unknowns.append(consistent_latent)
          
          if len(faces_to_replace) > 0:
            frame = swap_faces(opt, faces_to_replace, mats_to_replace, frame, model, closest_unknowns, spNorm, mask_model=mask_net)

      # Write the processed frame to the video writer
      out.write(frame)
      # Display the resulting frame
      cv2.imshow('Webcam Frame', frame)

      # Press 'q' to exit the loop
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break
    
  # Release video capture and writer objects
  cap.release()
  out.release()

if __name__ == "__main__":
  db_not_replaced = load_database('database_not_replaced.json')
  db_to_replace = load_database('database_to_replace.json')
  
  main(db_not_replaced=db_not_replaced, db_to_replace=db_to_replace)
