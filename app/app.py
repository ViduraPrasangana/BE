from flask import Flask, request, jsonify
import torch
from PIL import Image
from torchvision import transforms
from core.tokenizers import Tokenizer
from core.r2gen import R2GenModel
app = Flask(__name__)
device = "cpu"
transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
@app.route('/predict',methods=['POST'])
def predict():
    tokenizer = Tokenizer()
    model = R2GenModel(tokenizer)
    model = model.to(device)
    _resume_checkpoint(model)
    image = load_image()
    output = model(image, mode='sample')
    reports = model.tokenizer.decode_batch(output.cpu().numpy())
    return jsonify({'results':reports})

def _resume_checkpoint(self,model):
    resume_path = ""
    resume_path = str(resume_path)
    print("Loading checkpoint: {} ...".format(resume_path))
    checkpoint = torch.load(resume_path)
    model.load_state_dict(checkpoint['state_dict'],maploca)
    model.eval()
    print("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
    return model

def load_image():
    image_uri = "./data/1.png"
    image = Image.open(image_uri).convert('RGB')
    image = transform(image).unqueeze(0)
    return image