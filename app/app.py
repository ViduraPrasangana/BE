from flask import Flask, request, jsonify, json
import torch
from PIL import Image
from torchvision import transforms
from core.tokenizers import Tokenizer
from core.r2gen import R2GenModel
from torchinfo import summary
from PIL import Image
import base64
from io import BytesIO

app = Flask(__name__)
device = "cpu"
transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])

args = {
    "num_layers" : 3,
    "d_model" : 512,
    "d_ff" : 512,
    "num_heads" : 8,
    'dropout' : 0.1,
    "rm_num_slots" : 3,
    "rm_num_heads" : 8,
    "rm_d_model" : 512,
    "drop_prob_lm" : 0.5,
    "max_seq_length" : 60,
    "d_vf" : 1024,
    "d_model" : 512,
    "bos_idx" : 0,
    "eos_idx" : 0,
    "pad_idx" : 0,
    "use_bn"  : 0
}
class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)
args = Struct(**args)
model = None

@app.route('/predict',methods=['POST'])
def predict():
    if(model == None):
        initialize_model()
    data = request.data
    data = json.loads(data)
    image = load_image(data['image'])
    output = model(image, mode='sample')
    reports = model.tokenizer.decode_batch(output.cpu().numpy())
    return jsonify({'results':reports})

def initialize_model():
    global model
    tokenizer = Tokenizer()
    model = R2GenModel(args,tokenizer)
    model = model.to(device)
    model = _resume_checkpoint(model)
    return model

def _resume_checkpoint(model):
    resume_path = "./models/base_chexnet.pth"
    resume_path = str(resume_path)
    print("Loading checkpoint: {} ...".format(resume_path))
    checkpoint = torch.load(resume_path,map_location=torch.device(device))
    # summary(model, input_size=(1,3, 224, 224))
    a = model.load_state_dict(checkpoint['state_dict'],strict=False)
    print(a)
    model.eval()
    return model

def load_image(image):
    # image_uri = "./data/1.png"
    # image = Image.open(image_uri).convert('RGB')
    image = Image.open(BytesIO(base64.b64decode(image)))
    # image.save("re.jpg",format="jpeg")
    image = transform(image).unsqueeze(0)
    return image