# Few-Shot-Proto-TL
This is a repo for the paper titled Post-Stroke Hand Gesture Recognition via One-Shot Transfer Learning using Prototypical Networks

The research presented in this paper improves the classification accuracy of subject-independent models for hand-gesture recognition post-stroke by using prototypical networks for one-shot transfer learning from the new participant to improve model accuracy from subject-independent models. This is the first paper to propose the application of few-shot learning for adapting a generalized model to individual users. The proposed approach is contrasted against conventional transfer learning as well as subject-dependent and subject-independent classifiers, and evaluated on data collected from 20 stroke survivors performing seven distinct gestures.

![Fig2](https://github.com/HSarwat/Few-Shot-Proto-TL/assets/58105330/5fa9f65d-fbda-4cf8-b80b-bcb9cff2f637)
The extracted feature vector FN is fed into a fully connected neural network to generate embedding features. These features map each class prototype (G1, G2,... G7), obtained from the mean of the support set (s), to a position in the embedding space. The class for each new sample (Q) is chosen by using a distance function to identify the closest class prototype. The extracted feacture vector used for this study is available in "processing/Data/data.pkl", and the code can be run after modifying the used paths to the installed ones.

![Fig3](https://github.com/HSarwat/Few-Shot-Proto-TL/assets/58105330/7c427d51-bace-4afa-8a6e-5c878b769168)
Our proposed methodology (one-shot PN with 0.88s window size) was significantly more accurate than all other benchmark models (p < 0.05).

![Fig10](https://github.com/HSarwat/Few-Shot-Proto-TL/assets/58105330/d89925c8-c70a-426f-be1e-0fc54ff38b07)
Our proposed approach performed similarly to subject-dependent models.

A major problem in rehabilitation research is the lack of generalized models that can work well for different upeople. Current methods often struggle to adapt to the unique conditions of each person. Without these models, it’s hard to successfully implement wearable systems for in-home rehabilitation. Our research shows that when compared to subject-dependent classifiers, our approach consistently achieves similar results. This suggests that our method can effectively use large-scale models in new users without losing accuracy. This flexibility is valuable for rehabilitation research, where subject-specific data are limited
