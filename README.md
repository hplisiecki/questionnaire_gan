**Questionnaire GAN: Detecting & Classifying Fake Data in Online Surveys**

---

:star: **Overview:**
Welcome to the `Questionnaire GAN` repository! Dive into cutting-edge techniques that utilize Generative Adversarial Networks (GANs) to both simulate survey responses and unearth fake data in online questionnaires. By leveraging the power of GANs, this project aims to set a gold standard in ensuring the integrity of online survey data.

:computer: **Features:**

- **Survey Simulation via GAN:** Train a generator to craft artificial yet realistic survey responses.
- **Fake Data Detection:** Once the generator is trained, an overtrained discriminator emerges as a potent tool for pinpointing fake data in online responses. This pioneering approach offers a data-independent online fake response detection mechanism.
- **Advanced Classification:** The repository doesn't stop at fake data detection! Delve into generator functions that birthed an associated project focused on classifying non-ergodic survey responses.

:file_folder: **Repository Structure:**

- `detection_gan` folder: Houses code related to the training of the GAN for fake data detection.
- `simple_net` folder: Houses code related to a connected project aiming to detect non-ergodic data
- The rest of the code files were used to generate the required datasets, either from scratch when it comes to the ergodicity project or from online survey databases, which are used to train the GAN

:bulb: **Potential Applications:**

- Enhance the validity of online surveys by filtering out fake responses.
- Improve survey design by understanding and classifying non-standard responses.
- Research on data integrity in online platforms.

:wrench: **Get Started:**
Want to contribute, adapt, or learn from this project? Start by exploring the `detection_gan` folder for core functionalities and branch out to the generator functions to understand the classification of non-ergodic responses.

---

Join us in reshaping the future of online surveys and ensuring data validity! Feedback, contributions, and insights are always welcome. :rocket:

---
