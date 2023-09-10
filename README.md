# Questionnaire Gan

This repository houses the code that can be used to train a Generative Adversarial Network to simulate survey responses. 
The aim of this project was to design a novel way of detecting fake data in online questionnaires. By training the generator
to simulate artificial survey responses, and then overtraining the discriminator a data independent online fake response 
detector can be achieved. The code for this part of the project is housed at the "detection_gan" folder". Beyond that, the
repository also includes a couple of generator functions which were later on used for a connected project, that aimed
to create a classificator of non-ergodic survey responses.
