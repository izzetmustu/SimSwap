import cv2
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from models.models import create_model
from options.recognize_options_dataset import RecognizeOptionsDataset
from insightface_func.face_detect_crop_multi import Face_detect_crop
from util.reverse2original import reverse2wholeimage
import os
from util.norm import SpecificNorm
from parsing_model.model import BiSeNet
import json
from moviepy.editor import VideoFileClip

import onnxruntime as ort
ort.set_default_logger_severity(3)

def load_database(db_path:str=None):
  with open(db_path, 'r') as f:
    return json.load(f)
  
transformer_Arcface = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
  ])

def _totensor(array):
  tensor = torch.from_numpy(array)
  img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
  return img.float().div(255)

def cosine_similarity(embedding1, embedding2):
  dot_product = np.dot(embedding1, embedding2)
  norm1 = np.linalg.norm(embedding1)
  norm2 = np.linalg.norm(embedding2)
  return dot_product / (norm1 * norm2)
  
def get_latents_from_image(img:np.ndarray, opt:dict, app:Face_detect_crop, model):
  with torch.no_grad():
    img_align_crop_list, _ = app.get(img, opt.crop_size)
    img_latents = []
    
    for img_align_crop in img_align_crop_list:
      img_align_crop_pil = Image.fromarray(cv2.cvtColor(img_align_crop, cv2.COLOR_BGR2RGB))
      img_transformed = transformer_Arcface(img_align_crop_pil)
      img_id = img_transformed.view(-1, img_transformed.shape[0], img_transformed.shape[1], img_transformed.shape[2])
      img_id = img_id.cuda()
      img_id_downsample = F.interpolate(img_id, size=(112,112))
      latend_id = model.netArc(img_id_downsample)
      latend_id = F.normalize(latend_id, p=2, dim=1)
      img_latents.append(latend_id)
    
    return img_latents
    
def get_latent(face:np.ndarray, model):
  with torch.no_grad():
    img_align_crop_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
    img_transformed = transformer_Arcface(img_align_crop_pil)
    img_id = img_transformed.view(-1, img_transformed.shape[0], img_transformed.shape[1], img_transformed.shape[2])
    img_id = img_id.cuda()
    img_id_downsample = F.interpolate(img_id, size=(112,112))
    latend_id = model.netArc(img_id_downsample)
    latend_id = F.normalize(latend_id, p=2, dim=1)
  
  return latend_id

def find_closest(face_embedding, db:dict):
  best_match_name = None
  best_match_score = float('-inf')
  closest_latent = None
  best_index = None
  
  for name, db_embeddings in db.items():
    for index, db_embedding in enumerate(db_embeddings):
      score = cosine_similarity(face_embedding.cpu().numpy(), db_embedding)
      if score > best_match_score:
        best_match_score = score
        best_match_name = name
        closest_latent = db_embedding
        best_index = index
        
  return best_match_name, closest_latent, best_match_score, best_index

def swap_faces(opt, faces, mats, img, model, latents, spNorm, mask_model):
  # detect_results = None
  swap_result_list = []
  b_align_crop_tenor_list = []

  for b_align_crop, latend_id in zip(faces, latents):
    b_align_crop_tenor = _totensor(cv2.cvtColor(b_align_crop,cv2.COLOR_BGR2RGB))[None,...].cuda()
    swap_result = model(None, b_align_crop_tenor, latend_id, None, True)[0]
    swap_result_list.append(swap_result)
    b_align_crop_tenor_list.append(b_align_crop_tenor)

  return reverse2wholeimage(b_align_crop_tenor_list,swap_result_list, mats, opt.crop_size, img, None, \
      None, True, pasring_model =mask_model,use_mask=opt.use_mask, norm = spNorm)
    
def add_audio_to_video(original_video_path, video_without_audio_path, output_path):
  # Load original video with audio
  original_video = VideoFileClip(original_video_path)
  
  # Load the processed video (without audio)
  processed_video = VideoFileClip(video_without_audio_path)
  
  # Combine the audio from the original video with the processed video
  final_video = processed_video.set_audio(original_video.audio)
  
  # Write the final video with the original audio to the output file
  final_video.write_videofile(output_path, codec='libx264')
     
# Initialize a history dictionary to store the last used embedding for unknown faces
face_history = {}

# Function to compute the Euclidean distance between bounding boxes
def bbox_distance(bbox1, bbox2):
    center1 = [(bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2]  # center of bbox1
    center2 = [(bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2]  # center of bbox2
    return np.linalg.norm(np.array(center1) - np.array(center2))

# Update face history with bounding box correction
def update_face_history(face_embedding, closest_latent, bbox, threshold=0.4, position_tolerance=80):
    # Iterate through previous faces in history to find a match by both embedding and position
    for previous_bbox, previous_embedding in face_history.items():
        position_dist = bbox_distance(previous_bbox, bbox)
        # print(f"Position distance: {position_dist}")
        if position_dist < position_tolerance:  # Ensure bounding box is nearby
            return previous_embedding
            similarity = cosine_similarity(face_embedding.cpu().numpy(), previous_embedding.cpu().numpy())
            print(f"Similarity: {similarity}")
            # If similarity is high and the position is close, it's the same face
            if similarity > threshold:
                print(f"Consistent face detected based on proximity (distance={position_dist}) and similarity.")
                return previous_embedding

    # If no match is found, update the history with the new bounding box and embedding
    face_history[tuple(map(int, bbox))] = closest_latent
    return closest_latent
  
def main(db_not_replaced:dict, db_to_replace:dict):
  opt = RecognizeOptionsDataset().parse()
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
  app.prepare(ctx_id= 0, det_thresh=0.5, det_size=(224,224),mode=mode)

  #################### UPDATE THRESHOLD ####################
  threshold = 1.0
  
  videos = os.listdir(opt.video_path)
  videos = sorted(videos)
  
  num_videos = len(videos)
  for i, video in enumerate(videos):
    print(f"Processing video {i+1}/{num_videos}, {video}")
  
    cap = cv2.VideoCapture(os.path.join(opt.video_path, video))
    # Total frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    counter = 1
    
    # Get the frame width, height, and frames per second (fps)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    # print video properties
    print(f"Width: {width}, Height: {height}, FPS: {fps}")
    print(f"Total frames: {total_frames}")
    print(f"Total duration: {total_frames/fps} seconds")
    
    # VideoWriter to save processed video frames
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(os.path.join(opt.output_path, 'tmp.mp4'), fourcc, fps, (width, height))
    
    while cap.isOpened():
      ret, frame = cap.read()
      if not ret:
          break
      
      if counter % 100 == 0:
        print(f"Processing frame {counter}/{total_frames}")
      counter += 1

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
              consistent_latent = update_face_history(face_embedding, closest_unknown, bbox)
      
              faces_to_replace.append(face)
              mats_to_replace.append(mat)
              closest_unknowns.append(consistent_latent)
          
          if len(faces_to_replace) > 0:
            frame = swap_faces(opt, faces_to_replace, mats_to_replace, frame, model, closest_unknowns, spNorm, mask_model=mask_net)

      # Write the processed frame to the video writer
      out.write(frame)
      
    # Release video capture and writer objects
    cap.release()
    out.release()
    
    # Combine audio and the processed video
    add_audio_to_video(os.path.join(opt.video_path, video), os.path.join(opt.output_path, 'tmp.mp4'), os.path.join(opt.output_path, video))

if __name__ == "__main__":
  db_not_replaced = load_database('database_not_replaced.json')
  db_to_replace = load_database('database_to_replace.json')
  
  main(db_not_replaced=db_not_replaced, db_to_replace=db_to_replace)
