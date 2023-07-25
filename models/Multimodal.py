import torch
import torch.nn as nn

# Combine the Multiple modules into a single model
class EnsembleModel(nn.Module):

    def __init__(self, model1, model2):
        super().__init__()

        self.model1 = model1
        self.model2 = model2

        self.regressor = nn.Sequential(
        nn.Linear(256*2, 256*2),
        nn.Linear(256*2, 1)) 

    def forward(self, video_embed, audio_embed):
        video_embed = self.model1(video_embed)
        audio_embed = self.model2(audio_embed)
        output = torch.cat((video_embed, audio_embed), dim=1)
        output = self.regressor(output)
        return output


# Combine the Multiple modules into a single model
class EnsembleModelClassifier(nn.Module):

    def __init__(self, model1, model2):
        super().__init__()

        self.model1 = model1
        self.model2 = model2

        self.regressor = nn.Sequential(
        nn.Linear(256*2, 256*2),
        nn.ReLU(),
        nn.Linear(256*2, 4)
        ) 

    def forward(self, video_embed, audio_embed):
        video_embed = self.model1(video_embed)
        audio_embed = self.model2(audio_embed)
        output = torch.cat((video_embed, audio_embed), dim=1)
        output = self.regressor(output)
        output = nn.functional.softmax(output)
        # output = torch.argmax(output).view(-1, 1)
        # print(output)
        return output