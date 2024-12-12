from classification_app.model import ResNet_18
import torchvision.transforms as transforms
import torch
import cv2
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'

PATH = "../model/best_model_params.pt"
in_channels = 3
num_classes = 15
id2label= {0: 'Bean', 1: 'Bitter_Gourd', 2: 'Bottle_Gourd', 3: 'Brinjal', 4: 'Broccoli', 5: 'Cabbage', 6: 'Capsicum', 7: 'Carrot', 8: 'Cauliflower', 9: 'Cucumber', 10: 'Papaya', 11: 'Potato', 12: 'Pumpkin', 13: 'Radish', 14: 'Tomato'}
model = ResNet_18(3, 15)
state_dict = torch.load(PATH, weights_only=True,map_location="cpu")
model.load_state_dict(state_dict)


data_transforms = {
    'inference': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def inference(img_path):
    model.eval()
    img = Image.open(img_path)
    img = data_transforms['inference'](img)
    img = img.unsqueeze(0)
    img = img.to(device)
    outputs = model(img)
    class_prob = torch.softmax(outputs, dim=1)
    _, preds = torch.max(class_prob, 1)
    #print('Probability 7 %.2f%%' % (preds)  )
    print(f"preds {preds[0].tolist() , _[0].tolist()} , outputs {class_prob}")
    class_name = id2label[preds[0].tolist()]
    return class_name , _[0].tolist()

#inference("./samples/1016.jpg")