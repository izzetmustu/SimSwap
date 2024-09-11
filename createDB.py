import numpy as np
import cv2
import torch
from insightface_func.face_detect_crop_multi import Face_detect_crop
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import json
import os

transformer_Arcface = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

class NumpyFloatValuesEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)

class DatabaseCreator():
  def __init__(self, name='antelope', root='./insightface_func/models', netArc_checkpoint='./arcface_model/arcface_checkpoint.tar', det_size=(224,224), crop_size=512):
    self.name = name
    self.root = root
    self.netArc_checkpoint = netArc_checkpoint
    self.det_size = det_size
    self.crop_size = crop_size
    # Detection Model
    self.app = Face_detect_crop(name=self.name, root=self.root)
    self.app.prepare(ctx_id= 0, det_thresh=0.6, det_size=self.det_size,mode=None)
    # Recognition Model
    # Id network
    self.netArc = torch.load(self.netArc_checkpoint, map_location=torch.device("cpu"))
    self.netArc = self.netArc.cuda()
    self.netArc.eval()
    
  def createDB(self, people_dict, write_path=None):  # Example dict: {'Person1': ['path/to/image']}
    # Database of face embeddings
    with torch.no_grad():
      face_db = {}
      for person_name, images in people_dict.items():
        ids = []
        print(person_name)
        print(images)
        for image in images:
          img_a_whole = cv2.imread(image)
          img_a_align_crop, _ = self.app.get(img_a_whole, self.crop_size)
          img_a_align_crop_pil = Image.fromarray(cv2.cvtColor(img_a_align_crop[0],cv2.COLOR_BGR2RGB)) 
          img_a = transformer_Arcface(img_a_align_crop_pil)
          img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])
          # convert numpy to tensor
          img_id = img_id.cuda()
          #create latent id
          img_id_downsample = F.interpolate(img_id, size=(112,112))
          latend_id = self.netArc(img_id_downsample)
          latend_id = F.normalize(latend_id, p=2, dim=1)[0]
          
          cid = list(latend_id.cpu().numpy())
          ids.append(cid)
        face_db[person_name] = ids

      if(write_path):
        with open(write_path, 'w') as fp:
            json.dump(face_db, fp, cls=NumpyFloatValuesEncoder)
      
      return(face_db)
      
def create_dict_known(path):
  people = {}
  for person in os.listdir(path):
    if os.path.isdir(os.path.join(path, person)):
      people[person] = []
      for image in os.listdir(os.path.join(path, person)):
        people[person].append(os.path.join(path, person, image))
  return people    

def create_dict_unknown(path):
  people = {}
  images = os.listdir(path)
  for index, person in enumerate(images):
      people[str(index+1)] = [os.path.join(path, person)]
  return people     

def main():
  databasecreator = DatabaseCreator()
  # Known people database
  known_dict = create_dict_known('/mnt/storage1/izzet/datasets/known_people')
  db_known = databasecreator.createDB(known_dict, write_path='./database_not_replaced.json') 

  # People to be replaced database
  unknown_dict = create_dict_unknown('/mnt/storage1/izzet/datasets/5k')
  db_unknown = databasecreator.createDB(unknown_dict, write_path='./database_to_replace.json')

if __name__ == "__main__":
  main()