This is NLPIR course project on Fine-tuning Pre-trained models for Amharic Speech Emotion Recogntion.
The model is a fine-tuned version of Wev2Vec2.0 and HuBERT base & large vesrion on the ASED dataset.

To use fine-tuned Amharic SER model:

#  Wev2Vec2.0
from transformers import AutoProcessor, Wav2Vec2ForSpeechClassification

processor = AutoProcessor.from_pretrained("Gizachew/wev2vec-base960-agu-amharic")

model = Wav2Vec2ForSpeechClassification.from_pretrained("Gizachew/wev2vec-base960-agu-amharic")

Wev2Vec2.0-base: https://huggingface.co/Gizachew/wev2vec-base960-agu-amharic 

Wev2Vec2.0-large: https://huggingface.co/Gizachew/wev2vec-large960-agu-amharic 


# HuBERT
from transformers import AutoProcessor, HubertForSpeechClassification

processor = AutoProcessor.from_pretrained("Gizachew/hubert-agum960-amharic")

model = HubertForSpeechClassification.from_pretrained("Gizachew/hubert-agum960-amharic")

HuBERT-base: https://huggingface.co/Gizachew/hubert-base960-agu-amharic 

HuBERT-large: https://huggingface.co/Gizachew/hubert-agum960-amharic

Demo: https://huggingface.co/spaces/Gizachew/Amharic-SER

The orginal dataset used for this experiment is in this link: https://github.com/Ethio2021/ASED_V1 
