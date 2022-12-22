import numpy as np
import torch
from torchvision import transforms
import torch.nn.functional as nnf

class Detector():

    def __init__(self, checkpoint_path, min_confidence=0.9):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model = checkpoint
        self.model.eval()

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self.min_confidence = min_confidence

        # define the action array, one element is the action of one frame
        self.actions = {
            'FIVE': np.zeros(10), 
            'OK': np.full(10, 8), 
            'UP': [np.array([9, 9, 9, 6, 6, 6, 6, 6, 6, 6]),
                    np.array([9, 9, 9, 6, 6, 6]),
                    np.array([9, 9, 9, 9, 9, 6, 6, 6, 6, 6]),
                    np.array([9, 9, 9, 6, 6, 6, 6, 6])],
            'DOWN': [np.array([6, 6, 6, 9, 9, 9, 9, 9]),
                    np.array([6, 6, 6, 9, 9, 9, 9]),
                    np.array([6, 6, 6, 6, 9, 9, 9, 9, 9]),
                    np.array([6, 6, 6, 6, 9, 9, 9, 9])]}

    '''
    return -1 when no result else return label code from 0 to num_class
    '''
    def predict_landmarks(self, landmarks): 

        input = torch.unsqueeze(self.transforms(landmarks), 0).to(self.device)
        with torch.no_grad():
            ret = self.model(input)

        return_value = np.zeros(ret.shape[0])
        if torch.all(ret < 0):
            return -1, 0, return_value, return_value

        prob = nnf.softmax(ret / 2, dim=1)
        confidence = prob.max(1, keepdim=True)[0]
        confidence = confidence.cpu().numpy().copy().squeeze()
        if confidence < self.min_confidence:
            return -1, 0, return_value, return_value

        label_cd = ret.max(1, keepdim=True)[1]
        label_cd = label_cd.cpu().numpy().copy().squeeze()
        return label_cd, confidence, prob.cpu().numpy().copy().squeeze(), ret.cpu().numpy().copy().squeeze()

    '''
    detect moving direction, distance and speed
    '''
    def detect_properties(self, actions):
        None

    '''
    detect a combination of actions
    '''
    def detect_action(self, actions):

        for key, value in self.actions.items():
            if type(value) is list:
                for v in value:
                    ret = (actions[len(actions) - len(v):] == v)
                    if np.all(ret): return key
            elif type(value).__module__ == np.__name__:
                ret = (actions[len(actions) - len(value):] == value)
                if np.all(ret): return key

    '''
    detect map track
    '''
    def detect_map_track(self, actions):
        None        