# Emotion recognition
## Motivation
Nowadays emotion recognizing neural networks are becoming more and more popular. There are nine main emotions such as happy, sad, fear, anger, neutral, uncertain, surprise, disgust, contempt. In many business cases it can be applied. For instance in amusement park where AI can detect whether the person is sad or not and if it needs, can help people increase their happiness. Also in banking sphere where also AI conslultant may communicate with clients and help to make people happier or less nervous by dialog based on their emotions.
## Project description
Using neural network to recognize emotion from pictures.
## Project structure
`emotion_file.txt` - file with nine emotions which model can detect. `emotion-recognizing-neural-network.py` - file with implementation of the emotion recognizing model. `frozen_graph.pb` - frozen neural model. `training-neural-model.ipynb` - notebook with training the neural network
## Necessary addition
In `training-neural-model.ipynb` file we train the model to recognize emotions, but for detecting the faces you also have to download face-recognition library by using next command: `pip install face-recognition`. If you want to know more details about this fantastic library, [here](https://pypi.org/project/face-recognition/) is the link
## Possible imporovements
The emotion-recognizing model in this project has the accuracy approximately 50%. If you want to increase the accuracy, there are some tips. Firstly, you can put as input not only pictures but also audio into the model such as dialog, monolog or any emotional moment. Secondly, put several frames of video. Becasue it is important for the model to know what kind of events preceded and then it will be easier to define what emotions people feel.
