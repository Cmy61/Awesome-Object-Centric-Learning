# Awesome-Object-Centric-Learning
### MONet: Unsupervised Scene Decomposition and Representation - 22 Jan 2019

https://arxiv.org/pdf/1901.11390.pdf

**Task:** Implement unsupervised scene decomposition (based on semantics) ; obtain visual object representations.

**Background:** nowadays, most models are supervised + simple datasets.

**Introduction:** An architecture for learning segmentation and representation of image components. The model consists of a segmentation network and a variational autoencoder (VAE) trained in a concatenated manner.

**Method:** Cycle-attention network + autoregression + processing one object per network iteration.

<img src="https://github.com/Cmy61/Awesome-Object-Centric-Learning/blob/main/image/image-20230421214406584.png" alt="image-20230421214406584" width="70%" hight="70%"  style="zoom:50%;" />

* Attention Network

Input: Original image + scope (indicating the processing range, objects that have been previously attended or reconstructed are excluded in subsequent steps)

Output: Attention mask + updated scope=1 (softmax); no attention mask generated in the last step

* Component VAE

Input/Output: Original image + mask (modeling the masked regions) --> Generates reconstruction of a specific object

* End-to-end Training


**Dataset:** Non-trivial 3D scenes with varying numbers of objects (e.g., CLEVR) + Objects Room dataset + Multi-dSprites. 
**Limitation**: We haven't dealt with datasets that have increased visual complexity.

### TOWARDS CAUSAL GENERATIVE SCENE MODELS VIA COMPETITION OF EXPERTS-2020 4 27

https://arxiv.org/pdf/2004.12906.pdf

**Task:** Generating scene models

**Background:** Previous generative models lack the ability to capture the inherent composition and hierarchical nature of visual scenes.

**Introduction:** During training, experts compete to explain different parts of the scene, focusing on different object categories. The method is based on two main ideas: treating scenes as a hierarchical combination of (partial) depth-ordered objects, and using a set of generative models or experts to represent object classes separately.

**Architecture:** **Multiple generative models competing to generate images (objects).**

<img src="https://github.com/Cmy61/Awesome-Object-Centric-Learning/blob/main/image/image-20230513164516477.png" alt="image-20230513164516477" width="70%" hight="70%"  style="zoom:50%;" />

**ECON (Expert Competing Object Networks)** contains multiple experts (generative modules). Each expert includes:

* Attention Network: The attention network is used to select the regions of interest in the image. It calculates the probability of each region belonging to a specific object category, determining which image regions the generative module should focus on.
* Encoder: The encoder maps the image within the selected regions to a latent encoding representation, denoted as z. The encoder learns to transform image information into a representation in the latent space for subsequent generation and inference.
* Decoder: The decoder receives the latent encoding z and generates the reconstruction of the object and its unmasked shape mt. The decoder is responsible for generating specific objects along with their shape information.

The optimal generative module is selected through a **competitive mechanism:**

* In each inference step t = T, ..., 1, all generative modules kt are applied to the current input (x, st).
* The winner ^kt is obtained through a competitive objective function.
* The reconstructed scene components are generated using the winning generative module ˆkt.
* The winning generative module is updated using gradient steps.

**Dataset:** Experiments were conducted on synthetic data composed of colored 2D objects  (triangles, squares, and circles) arranged with different occlusions. 

### GENESIS: GENERATIVE SCENE INFERENCE AND SAMPLING WITH OBJECT-CENTRIC LATENT REPRESENTATIONS - 23 Nov 2020

https://arxiv.org/pdf/1907.13052.pdf

**Task:** Scene generation model.
**Introduction:** GENESIS is the first object-centric rendering 3D scene generation model and it is capable of decomposing and generating scenes by capturing the relationships between scene components.

**Method:** GENESIS generative model allows parallel encoding and decoding.

<img src="https://github.com/Cmy61/Awesome-Object-Centric-Learning/blob/main/image/image-20230422155141489.png" alt="image-20230422155141489" width="70%" hight="70%"  style="zoom:50%;" />


**Dataset:** Colorful Multi-dSprites, GQN dataset, ShapeStacks. **Task:** Given a starting image and a target image, search for the best sequence of actions (how to move objects) to achieve the target image.

### Object-centric Forward Modeling for Model Predictive Control-2020

**Task**: Given the start image and the target image, search for the best sequence of actions (how to move the object) to get the target image
**Introduction:** A method for learning object-centric forward models that can be used for planning a sequence of actions to achieve long-term goals. Each object has an explicit spatial position and implicit visual features, and learn to use random interactive data to model action effects.

**Method:** **Obtain objects, predict the next representation (continuously correcting) based on the action it will take.**

<img src="https://github.com/Cmy61/Awesome-Object-Centric-Learning/blob/main/image/image-20230501181027189.png" alt="image-20230501181027189" width="70%" hight="70%" style="zoom:50%;" />

 **Object-centric representation:**

* Object-level representation (xtn): Each object is represented as a tuple combining position (btn) and visual feature (ftn).

**Object-centric forward model:** It predicts the representation of each object at the next time step t+1, based on the current object representation xtn and the action to be executed at+1, denoted as {xt+1 n}. Equation:  --> Achieve prediction.

* Forward model p: Predict future states (multiple iterations), Interaction Network model.
* Decoder: To further regularize features and encode meaningful visual information, decodes into pixels (used for supervision and to regulate the quality of feature encoding).
* Planning through forward models.
* Robust closed-loop control through correction modeling: To prevent significant deviations in long-term planning, introduce a correction model that updates the predicted positions based on new observed images.

**Dataset:** Synthetic environments in MuJoCo and real Sawyer robot (as the paper addresses how to make the robot prepare and adjust objects to reach a target position).


### SPACE:UNSUPERVISED OBJECT-ORIENTED SCENE - 15 Mar 2020

https://arxiv.org/pdf/2001.02407.pdf

**Task:** Multi-object scene decomposition (scene representation).

**Background:** Previous unsupervised learning methods for object-centric scene representation had limited scalability and faced obstacles in modeling real-world scenes.

**Introduction:** A generative latent variable model called SPACE is proposed, which provides a unified probabilistic modeling framework that combines the best spatial attention and scene mixture methods.

**Method:** **SPACE = Foreground Module + Background Module; Foreground and background are learned separately, and then combined to form the entire image.**

<img src="https://github.com/Cmy61/Awesome-Object-Centric-Learning/blob/main/image/image-20230421205602861.png" alt="image-20230421205602861" width="70%" hight="70%"  style="zoom:50%;" />

* Foreground Module: Models the foreground as structured latent variables (dividing the image into multiple cells, each cell modeling an object in the scene, each cell associated with a set of latent variables), which are used to compute foreground image components and finally combine them into an average foreground image.
* Background Module: Models the mixture probability πk and RGB distribution separately to model background components.

**Dataset:** Atari, 3D-Rooms 

### RELATE: Physically Plausible Multi-Object Scene Synthesis Using Structured Latent Spaces-2020

**Task:** Physically synthesizing multi-object scenes (generating images from given scene descriptions such as object positions, shapes, and appearances)+target scene editing+unsupervised model. 
**Background:** Image generation is typically achieved through Generative Adversarial Networks (GANs), where the generated images are realistic but **the parameterized random vectors behind them are not interpretable**.

**Introduction:** A model for learning to generate physically plausible scenes and videos with multiple interacting objects. RELATE combines object-centric GAN formulation with a model that explicitly specifies the relationships between individual objects. RELATE emphasizes the relationships between individual objects. The paper introduces a spatial relational network to enhance object interactions.

**Method:**

<img src="https://github.com/Cmy61/Awesome-Object-Centric-Learning/blob/main/image/image-20230425114833812.png" alt="image-20230425114833812" width="70%" hight="70%"  style="zoom:50%;" />

* Interaction Module: Computes physically plausible relationships between objects and the relationships between objects and the background, considering spatial correlations to make the generated images more realistic.
* Scene Synthesis and Rendering Module: Samples foreground and background, aggregates them into a tensor, and generates images.
  * Appearance Parameter Sampling: Random appearance parameter sampling Z is performed for each foreground object and the background, assuming that the appearances of different objects are independent of each other.
  * Appearance Mapping: Each sampled appearance parameter is mapped to a tensor Ψ using two separate decoder networks.
  * Pose Parameters (Translation Parameters): To obtain the positions of foreground objects, RELATE also samples a 2D translation parameter θ for each foreground object, representing the object's position in the scene, which is geometrically interpretable.
  * Scene Composition: All foreground objects and background objects are aggregated into a scene tensor W, which is composed into an overall scene tensor using element-wise maximum (or sum) pooling, denoted as scene tensor W(θ,Z).
  * Scene Rendering: Uses a decoder network to render the synthesized scene tensor into an image.
* Modeling Relationships in Scene Composition: **Captures the relational information between objects to achieve image generation.**
  * The model samples a set of K independently and identically distributed translation vectors ˆΘ.
  * The sampled vectors are corrected through a correction network (Θ := Γ( ˆΘ, Z)) to obtain the corrected translation vectors Θ. (By using the correction network Γ, the pose parameters θk of objects can be corrected based on the pose parameters and appearance parameters of other objects, thereby establishing correlations between objects.)
* Applications:
  * Scenes with a natural order (a special case of the model): Modifying parameters.
  * Modeling dynamic scenes.
* Training Objective: The training objective of the model consists of two high-fidelity losses and a structural loss + a position regressor network.

**Dataset:** BALLSINBOWL, CLEVR (cluttered desktop), ShapeStacks (block stacking), REALTRAFFIC

**Applications:** RELATE has the ability to change the background and appearance of individual objects. RELATE can also modify the position of individual objects.

**Limitations:** The model is highly sensitive to the camera's perspective range in the scene, and it cannot accurately capture the variations in appearance introduced by significant changes in the viewpoint throughout a sequence.

### Object-Centric Image Generation from Layouts-2020.12.3

https://arxiv.org/pdf/2003.07449.pdf

**Task:** Generating complex scenes with multiple objects.

**Introduction: Layout to image generation with Object-Centric GAN (OC-GAN):** This approach relies on a novel Scene Graph Similarity Module (SGSM) in the image generation process.

**Problem addressed:** The OC-GAN architecture addresses two issues in previous architectures: (1) generated fake objects lacking corresponding bounding boxes in the layout, and (2) overlapping bounding boxes in the layout leading to merged objects in the generated images.

<img src="https://github.com/Cmy61/Awesome-Object-Centric-Learning/blob/main/image/image-20230427212441509.png" alt="image-20230427212441509" width="50%" hight="50%"  style="zoom:50%;" />

**Knowledge:**

* Scene Graph Similarity Module (SGSM): Improves the fidelity of generated image layouts. The SGSM module calculates the similarity between the scene graph and the generated image, providing fine-grained matching-based supervision between the positional scene graph and the generated image.

<img src="https://github.com/Cmy61/Awesome-Object-Centric-Learning/blob/main/image/image-20230427215018552.png" alt="image-20230427215018552" width="30%" hight="30%"  style="zoom:50%;" />

* Instance-aware conditioning: Helps the model map overlapping conditional semantic masks to individual object instances by introducing instance boundary information. This makes it easier for the model to distinguish unique object instances.

<img src="https://github.com/Cmy61/Awesome-Object-Centric-Learning/blob/main/image/image-20230429205537094.png" alt="image-20230429205537094" width="30%" hight="30%"  style="zoom:50%;" />

**Method: GAN with Layout as a Cue**

* Model Architecture: The OC-GAN model is based on the GAN framework.

  <img src="https://github.com/Cmy61/Awesome-Object-Centric-Learning/blob/main/image/image-20230427214526498.png" alt="image-20230427214526498" width="70%" hight="70%"  style="zoom:50%;" />

  * The Generator module generates images conditioned on the ground truth layout. It is constructed based on a classical residual architecture.
  * The Discriminator predicts whether the input image is real or generated. The Discriminator has an additional component that distinguishes the objects present in the input image blocks corresponding to the object bounding boxes in the ground reality layout.
    * Object Discriminator: Determines if the objects in the generated image are similar to those in the real image.
    * Patch Discriminator: Determines if the local regions of the input image are consistent with the layout in the real image.

**Dataset:** COCO-Stuff, Visual Genome.

### SIMONe: View-Invariant, Temporally-Abstracted Object Representations via Unsupervised Video Decomposition-2021

https://arxiv.org/pdf/2106.03849.pdf

**Task:** View synthesis and instance segmentation, learning object representation, attributes

**Background:** When inferring scene structure and features, it is necessary to simultaneously estimate the agent's position/viewpoint information. These two variables jointly affect the agent's observation results, making simultaneous inference a challenging problem.

**Introduction:** An unsupervised variational method is proposed to solve this problem. By leveraging shared structures existing across different scenes, the model learns to infer (separate) two sets of latent representations (separating scene structure from viewpoint information) from only RGB video input: a set of "object" latent representations corresponding to time-invariant object-level content of the scene, and a set of "frame" latent representations corresponding to globally varying elements over time, such as viewpoint.

<img src="https://github.com/Cmy61/Awesome-Object-Centric-Learning/blob/main/image/image-20230511173729904.png" alt="image-20230511173729904" width="50%" hight="50%"  style="zoom:50%;" />

As shown in the figure: by combining the latent representations of specific combinations of object appearances and frames from different sequences, it is possible to generate rendering results with different viewpoints but consistent scenes (as observed in the image on the right, revealing changes in camera pose and lighting).

**Method:** Separate scene structure from viewpoint information and reconstruct the image.

<img src="https://github.com/Cmy61/Awesome-Object-Centric-Learning/blob/main/image/image-20230511174058205.png" alt="image-20230511174058205" width="50%" hight="50%"  style="zoom:50%;" />

* Latent structure and generation process

  * Latent structure: The model uses a set of object latent variables (O) and a set of frame latent variables (F) to represent the scene. The object latent variables remain constant throughout the sequence, while the frame latent variables capture time-varying information.

  * Generation process: Use latent vectors to generate images.

Decoder: We independently decode each pixel, and the pixel-level decoder takes sampled latent variables, pixel coordinates, and time steps as inputs. The decoder's architecture can be MLP or 1x1 CNN.

**Dataset**: Objects Room 9 + CATER (moving camera) + Playroom

### ROOTS: Object-Centric Representation and Rendering of 3D Scenes-2021

https://arxiv.org/pdf/2006.06130.pdf

**Task:** Learn to build modular and compositional 3D object models from partial scene images (observing only partial images of the scene and learning how to construct complete 3D scenes)

**Introduction:** In this paper, a probabilistic generative model is proposed to learn to build modular and compositional 3D object models from partial observations of multi-object scenes. This is achieved through a novel nested autoencoder architecture.

**Method:** Learn and infer 3D center coordinates, infer object-level 3D appearance representation, and generate 3D images from 2D images captured from multiple viewpoints.

<img src="https://github.com/Cmy61/Awesome-Object-Centric-Learning/blob/main/image/image-20230424212320618.png" alt="image-20230424212320618" width="70%" hight="70%"  style="zoom:50%;" />

* Encoder:

  Extract object regions from each scene image and group them into object-level contexts. The key idea is to first infer the center position of each object in 3D coordinates. The encoder takes 2D images from multiple viewpoints as input.

  - Scene encoder: Obtain scene representation. The model encodes the contextual set into the 3D spatial structure of the scene and infers the 3D positions of each object in 3D space.
    - Input: Context observation
    - Output: Scene representation ψ

  - Attention grouping: Identify the image regions corresponding to the same object in different observation images.

    ψ is reshaped into a feature map in 3D space, from which the 3D center positions of each object are inferred and the image regions of each object are recognized across viewpoints.

  - Object-level GQN: Infer object-level 3D appearance representation zwhat.

* Decoder：Decode into complete 3D images.
  * Object renderer: For each object n, given its 3D appearance representation zwhat n and a query viewpoint vq, ROOTS can generate a 4-channel (RGB+Mask) image that describes the 2D appearance of the object from the vq viewpoint.
  * Scene synthesizer: Combine the image layers of each object to generate a complete scene image (3D image).

**Applications:**

* Since object position and appearance are separated in the learned object model, manipulating the latent position allows us to **move objects** without changing other factors (such as object appearance).
* Composability. Once the object model is learned, it can be **reconfigured** to form new scenes beyond the training distribution.
* Object model. By applying the object renderer to zwhat n and a set of query viewpoints, the object model learned in Figure 4A is further visualized.
* Scene generation. Similar to GQN, ROOTS can generate target observation results for a given scene from arbitrary query viewpoints.

**Dataset:** ShapeNet, MSM


### Self-supervised Video Object Segmentation by Motion Grouping-2021

https://arxiv.org/pdf/2104.07658.pdf

**Task:** (Single moving) object segmentation (in videos) + self-supervised + motion (optical flow)

**Introduction:** The system is able to segment objects by leveraging motion cues (i.e., motion segmentation). To achieve this, a simple variant of Transformer is introduced to segment optical flow frames into primary objects and the background, and it is self-supervised.

**Method:** (1) A CNN encoder for extracting compact feature representations, (2) an iterative binding module with learnable queries that acts similar to soft clustering, assigning each pixel to a motion group, (3) a CNN decoder that decodes each query individually into full-resolution layer outputs.

<img src="https://github.com/Cmy61/Awesome-Object-Centric-Learning/blob/main/image/image-20230423155438041.png" alt="image-20230423155438041" width="70%" hight="70%"  style="zoom:50%;" />

* CNN encoder:
  * Input: Precomputed optical flow between two frames
  * Output: Lower-resolution feature map
* Iterative binding:
  * Input: Feature map + two learnable query slots (representing foreground and background)
  * Output: Updated query slots
  * Purpose: Group image regions into individual entities, where pixels moving at the same rate in the same direction should be grouped together (grouping pixel regions with similar motion into a single entity to identify moving objects)
* CNN decoder
  * Input: Slot vectors broadcasted onto a 2D grid
  * Output: Outputs at the original resolution (including (unnormalized) single-channel alpha mask and reconstructed flow field)

**Limitations:**

* Firstly, existing benchmarks are mainly limited to motion segmentation into foreground and background, so the researchers chose to use two slots in this paper (equivalent to recognizing only one object).
* Secondly, only motion (optical flow) was explored as input, which significantly limits the model's ability to segment objects in the absence of motion information or in incomplete flows ; however, the self-supervised video object segmentation objective also applies to two-stream methods, so RGB can be incorporated.
* Thirdly, when the optical flow has noise or low quality, the current method may fail; in such cases, joint optimization of the flow refinement and segmentation can be a possible way forward.
* Motion segmentation in real-world scenarios, such as predator or prey, may require fast processing. The model runs at over 80fps at low resolution (possibly sacrificing some accuracy).

**Dataset**: MoCA, DAVIS2016, SegTrackv2, FBMS59


### Unsupervised Object-Level Representation Learning from Scene Images-2021

https://arxiv.org/pdf/2106.11952.pdf

**Background:** The success of contrastive self-supervised learning heavily relies on the object-centric prior provided by ImageNet, where different augmented views of the same image correspond to the same object. However, when pretraining on more complex scene images with many objects, the results are not as effective.

<img src="https://github.com/Cmy61/Awesome-Object-Centric-Learning/blob/main/image/image-20230505205043501.png" alt="image-20230505205043501" width="70%" hight="70%"  style="zoom:50%;" />

**Introduction:** This paper proposes a multi-stage framework for unsupervised object-level representation learning, leveraging image-level self-supervised pretraining as a prior for discovering object-level semantic correspondences in scene images. Specifically, it first utilizes an unsupervised region proposal algorithm to extract potential object-based regions in the scene images. Then, a region correspondence generation scheme is proposed to discover corresponding object instances for the proposed regions in the embedding space using a pretrained model from image-level contrastive learning. Finally, the obtained object-instance pairs are used to construct positive sample pairs for object-level representation learning.

**Applications:** Several cross-image object-instance pairs and visual correspondences are discovered, with a focus on improving representation learning through high-quality correspondences.

<img src="https://github.com/Cmy61/Awesome-Object-Centric-Learning/blob/main/image/image-20230505205410754.png" alt="image-20230505205410754" width="70%" hight="70%"  style="zoom:50%;" />

**Method:** The ORL extends the existing image-level contrastive learning framework to the object level by leveraging instance discrimination priors. The process involves several contrastive learning modules in Stage 1 and Stage 3.

* Image-level pretraining: Obtains an unsupervised pretrained model from image-level tasks to learn global information and visual features of images.
* Correspondence discovery: Utilizes the pretrained model to find other images in the training set most similar to each image, forming image pairs. Then, potential object regions of interest (RoIs) that may contain objects are generated through unsupervised region proposal algorithms.
* Object-level pretraining: The BYOL framework performs object-level representation learning. Further pretraining is performed on the RoIs using image pairs to learn more semantically and object-related representations. (The aim is to have similar objects, such as cars, closer in the embedding space for better learning. )

### GENESIS-V2: Inferring Unordered Object Representations without Iterative Refinement-jan 2022

https://arxiv.org/pdf/2104.09958.pdf

**Task**:The tasks involved are unsupervised image segmentation and object-centric scene generation, reasoning about discrete objects in the environment, and predicting or imagining a set of object behaviors.
**Background:** Current methods are limited to visually less complex simulated and real-world datasets. Additionally, object representation is often inferred using RNN, which doesn't scale well to large images with potentially many objects. It also requires prior initialization of a fixed number of object slots.

**Introduction:** This work proposes an embedding-based approach similar to iterative refinement but without the need for prior initialization of a fixed number of clusters. It can infer a variable number of object representations without using RNN or iterative optimization (using IC-SBP method).

**Method:** The model is based on a probabilistic graphical model with an autoencoder. +The object representation is based on IC-SBP (with a focus on IC-SBP).

<img src="https://github.com/Cmy61/Awesome-Object-Centric-Learning/blob/main/image/image-20230422155141489.png" alt="image-20230422155141489" width="70%" hight="70%"  style="zoom:50%;" />

IC-SBP clusters pixel embeddings into a variable number of soft attention masks. The algorithm involves sampling pixel locations not yet assigned to clusters, creating soft or hard clusters based on the distances between the embedding of the selected pixel location and all other pixel embeddings, and repeating this process until all pixels are explained or a stopping condition is reached. The output of the algorithm is a set of normalized attention masks (clusters) for k objects. **IC-SBP differs from iterative refinement as it doesn't require prior initialization of a fixed number of clusters.**

<img src="https://github.com/Cmy61/Awesome-Object-Centric-Learning/blob/main/image/image-20230503220254183.png" alt="image-20230503220254183" width="70%" hight="70%"  style="zoom:50%;" />

**Dataset:** ObjectsRoom, ShapeStacks, Sketchy, MIT-Princeton Amazon Picking Challenge (APC) 2016 object segmentation dataset. 


### ILLITERATE DALL-E LEARNS TO COMPOSE-2022-mar-14

https://arxiv.org/pdf/2110.11405.pdf

**Task**:The tasks include systematizing zero-shot image generation without text, generating complex images composed of multiple objects, and constructing novel scenes, synthetic generation, and image reconstruction.
**Background:** DALL·E requires a dataset of text-image pairs, and composability is provided by the text. In contrast, object-centric representation models like Slot Attention can learn compositional representations without text prompts. However, unlike DALL·E, their systematic generalization capability for zero-shot image generation is very limited.

**Introduction:** A simple but novel slot-based autoencoder architecture called SLATE is proposed to combine the strengths of both models: learning object-centric representations and allowing systematic generalization in zero-shot image generation without text. Therefore, this model can also be seen as a text-blind DALL·E model. It is unsupervised and does not require annotations.

**Knowledge:** DALL·E has recently demonstrated the capability of systematically generalizing zero-shot image generation. Trained on a dataset of text-image pairs, it can generate plausible images even from unfamiliar text prompts such as "avocado chair" or "lettuce hedgehog," which is a form of systematic generalization in the text-to-image domain.

<img src="https://github.com/Cmy61/Awesome-Object-Centric-Learning/blob/main/image/image-20230429215214103.png" alt="image-20230429215214103" width="70%" hight="70%"  style="zoom:50%;" />

DALL-E: Input text units serve as composable units, and the generated images exhibit consistency.

Slot Attention: Object slots serve as composable units, but the generated images lack consistency.

SLATE: Like Slot Attention, our model is not supervised based on text, and like DALL·E, it generates novel image compositions with global consistency.

**Method:**

**Using a Transformer as our image decoder while replacing text prompts with slot prompts extracted from concepts** to enhance the use of compositional biases in the image decoder learned from a library of images constructed from a given set.

* Using dVAE to obtain image tokens

  * Purpose: To downsample high-resolution images (converting high-dimensional representation to low-dimensional), making Transformer training more efficient.
  * Steps: Splitting input image into patches xi -> xi as input to the encoder, obtaining log probability distribution oi -> sampling of relaxed one-hot encoding zsoft i (tokens) for patch i -> obtaining patch reconstruction through the decoder.

  <img src="https://github.com/Cmy61/Awesome-Object-Centric-Learning/blob/main/image/image-20230512203953486.png" alt="image-20230512203953486" width="50%" hight="50%"  style="zoom:50%;" />

* Inference of object slots:

  * Purpose: Infer the position and content information of objects from the input image to generate object slots.
  * Steps: Mapping tokens zi to embedding vectors using a dictionary -> Fusion of patch content and position information to obtain ui (adding the embedding vector of each patch with the corresponding position embedding vector, ui contains content and position information of the patch) -> Passing ui as input to the slot attention to obtain slots S and attention maps A.

  <img src="https://github.com/Cmy61/Awesome-Object-Centric-Learning/blob/main/image/image-20230512204630726.png" alt="image-20230512204630726" width="50%" hight="50%"  style="zoom:50%;" />

* Reconstruction using Transformer

  * Reconstructing the input image
  * Obtaining S, ˆoi -> obtaining estimated values of DVAE tokens ˆzi -> obtaining reconstructed values of image patches ˆxi through the decoder gθ -> combining ˆxi to form the reconstructed image ˆx

  <img src="https://github.com/Cmy61/Awesome-Object-Centric-Learning/blob/main/image/image-20230512205336070.png" alt="image-20230512205336070" width="50%" hight="50%"  style="zoom:50%;" />

**Dataset:** CLEVR-Mirror (developed from the CLEVR dataset), Shapestacks, Bitmoji, Textured MNIST, CLEVRTex, and CelebA.


### Simple Unsupervised Object-Centric Learning for Complex and Naturalistic Videos-2022-may

https://arxiv.org/pdf/2205.14065.pdf

**STEVE: Video Slot Transformer**

**Task:** Unsupervised handling of various complex and natural videos.

**Background:** Previous works have been proven to be applicable only to toy or synthetic images_videos, or even if they can handle complex images_videos, they require some supervision on the initial frames, such as optical flow or annotations.

**Introduction:** In this paper, STEVE is proposed, an unsupervised model for object-centric learning in videos. This is achieved without adding too much complexity to the model architecture or introducing new objectives or weak supervision, primarily leveraging a **Transformer-based slot decoder**.

Preliminaries: Two Decoders

* Hybrid Decoder: In this approach, the decoder decodes each slot separately using decoding functions gRGB θ and gmask θ to obtain the object image x and alpha mask m, respectively. Then, the decoded target images are weighted summed to obtain the complete image, and these decoders are implemented using CNNs. The key limitation is that it has never been successful in handling scenes with high visual complexity, such as natural images.
* Autoregressive Slot Transformer Decoder (SLATE): It is argued that the hybrid decoder severely restricts the interaction between slots and the quality of reconstruction, and a powerful autoregressive decoder based on Transformer conditioned on slots should be used. **(Used in this model)**

**Method:**

Three main components: ① Image Encoder based on CNN ② Recurrent Slot Encoder, which updates slot representations using a recurrent neural network (RNN) ③ Slot Transformer **Decoder** (SLATE).

* Image Encoder: Obtains feature representations based on frames.
* Recurrent Slot Encoder: Takes the slot representation st-1 from the previous time step and the feature map xt of the current frame as input and updates the slot representation from st−1 to st. (Rough process)
* Slot Transformer Decoder: Reconstructs the current frame xt using the slot representation st. (By learning to autoregressively predict a sequence of discrete tokens in a frame.) For this reconstruction, each frame xt is treated as a sequence of discrete labels provided by a discrete VAE encoder. Given the slot st, the slot transformer decoder learns to autoregressively predict this sequence of labels by minimizing the cross-entropy loss.

<img src="https://github.com/Cmy61/Awesome-Object-Centric-Learning/blob/main/image/image-20230524191929722.png" alt="image-20230524191929722" width="50%" hight="50%"  style="zoom:50%;" />

<img src="https://github.com/Cmy61/Awesome-Object-Centric-Learning/blob/main/image/image-20230502205847839.png" alt="image-20230502205847839" width="50%" hight="50%"  style="zoom:50%;" />


Training: <img src="https://github.com/Cmy61/Awesome-Object-Centric-Learning/blob/main/image/image-20230502210105114.png" alt="image-20230502210105114" width="20%" hight="20%"  style="zoom:50%;" /> (Cross-entropy loss (CE) is used to train the slot-transformer decoder, and the discrete VAE (dvae) is used to train the discrete encoder and decoder for better slot representations.)

Datasets: CATER, CATERTex, MOVi-Solid, MOVi-Tex, MOVi-D, and MOVi-E


### ROBUST AND CONTROLLABLE OBJECT-CENTRIC LEARNING THROUGH ENERGY-BASED MODELS-oct-2022

https://arxiv.org/pdf/2210.05519.pdf

**Task:** Object Representation

**Introduction:** EGO is proposed as a method to learn object-centric representations using an energy-based model. By utilizing off-the-shelf self-attention blocks in Transformers to form permutation-invariant energy functions, we can infer object-centric latent variables using gradient-based MCMC methods, where permutation equivariance is automatically ensured.

<img src="https://github.com/Cmy61/Awesome-Object-Centric-Learning/blob/main/image/image-20230504202911236.png" alt="image-20230504202911236" width="70%" hight="70%"  style="zoom:50%;" />

**Method:** The goal of this method is to learn a mapping from visual observations x to a set of vectors {zk} that describe objects. We adopt an encoder-decoder architecture, where our EGO module serves as the encoder, transforming unstructured observations into structured object representations. EGO is actually designed to learn a mapping relationship between images and object representations.

* Energy Function: Used to measure the similarity or dissimilarity between the given observation data and latent variables. It takes the observation x and a single latent variable zk as inputs and outputs a scalar energy value that quantifies the confidence of zk representing the presence of a target object in the visual scene x. (This energy function is learned by a neural network and needs continuous learning.)
* Gradient-based MCMC Sampling: In order to infer the object-centric latent variable set z from the input x, we initialize z0 randomly from a simple prior distribution and iteratively update the latent variables.
* EGO uses MCMC to obtain latent variables z and then calculates the energy function between x and z.

**Specific Steps: Randomly initialize latent representation -> Update latent representation using energy function and gradient-based MCMC sampling method (Zt+1 involves Zt) -> Obtain zt+1 latent representation -> Iterate in a loop**

<img src="https://github.com/Cmy61/Awesome-Object-Centric-Learning/blob/main/image/image-20230511213742520.png" alt="image-20230511213742520" width="70%" hight="70%"  style="zoom:50%;" />

**Datasets:** CLEVR (Johnson et al., 2017), MultidSprites (Matthey et al., 2017), Tetrominoes (Greff et al., 2019), CLEVR-6


### Object-centric Learning with Cyclic Walks between Parts and Whole-2023 feb

https://arxiv.org/pdf/2302.08023.pdf

**Introduction:** We propose a cyclic walk between perceptual features extracted from CNN or Transformers and object entities. First, slot attention is used to obtain slot representations. Then, based on pairwise similarity between perceptual features (referred to as "parts") and slot-bound object representations (referred to as "wholes"), entity-feature correspondences are established along high transition probabilities. **Without a decoder, we propose cyclic walks on static images. Slot representations -> Cyclic Walk** Unsupervised.

<img src="https://github.com/Cmy61/Awesome-Object-Centric-Learning/blob/main/image/image-20230504215845974.png" alt="image-20230504215845974" width="40%" hight="40%"  style="zoom:50%;" />

Cyclic Walks:

<img src="https://github.com/Cmy61/Awesome-Object-Centric-Learning/blob/main/image/image-20230505200203629.png" alt="image-20230505200203629" width="60%" hight="60%"  style="zoom:50%;" />

("Parts" refer to feature vectors of non-overlapping image patches extracted from the input image, while "wholes" refer to object-centric representations.) (The cyclic walks process is similar to the interaction between wholes and parts in the part-whole theory, allowing the model to learn more accurate and robust representations of entity features and their correspondences.)

**Knowledge:**

* Pretrained self-supervised visual transformer DINO: Extracts feature vectors x from images, where each image is processed into non-overlapping patches, and each patch is projected into a feature embedding.

* Slot attention: Obtains slot representations of objects, denoted as ^s.

**Conventional** object-centric learning models use either decoder-based or transformer-based decoders to decode images from slots. The training objective of the model is to minimize the mean squared error loss between the decoder output and the original image at the feature or pixel level.

* Contrastive Random Walks: a and b are feature maps extracted from video frames using CNN or Transformers. The adjacency matrix is calculated (This matrix is used to construct a directed graph where each node corresponds to a feature. Exploring on the graph using random walks can learn relationships between features.)
* <img src="https://github.com/Cmy61/Awesome-Object-Centric-Learning/blob/main/image/image-20230505201622589.png" alt="image-20230505201622589" width="50%" hight="50%"  style="zoom:50%;" />


**Method: slot attention + static walks (updating slot representations iteratively)**

* Image feature extractor DINO + Obtain slot representations using SLOT ATTENTION

* Whole-Parts-Whole Cyclic Walks introduce two directions of cyclic walks: (a) from wholes to parts and back to wholes (W-P-W walk) and (b) from parts to wholes and back to parts (P-W-P walk), **to learn more accurate feature representations.**

  * W-P-W Cyclic Walks: As a supervisory signal for object-centric representation learning, x is parts, and ^s is wholes.
  * <img src="https://github.com/Cmy61/Awesome-Object-Centric-Learning/blob/main/image/image-20230505202512586.png" alt="image-20230505202512586" width="40%" hight="40%"  style="zoom:50%;" />

Loss term: <img src="https://github.com/Cmy61/Awesome-Object-Centric-Learning/blob/main/image/image-20230505202546333.png" alt="image-20230505202546333" width="20%" hight="20%"  style="zoom:50%;" />

* P-W-P Cyclic Walks: While W-P-W walks enhance the diversity of slot bases, there exists an ill-posed case where a finite set of slot bases cannot cover all semantic content of an image.

<img src="https://github.com/Cmy61/Awesome-Object-Centric-Learning/blob/main/image/image-20230505202832391.png" alt="image-20230505202832391" width="20%" hight="20%"  style="zoom:50%;" />

 <img src="https://github.com/Cmy61/Awesome-Object-Centric-Learning/blob/main/image/image-20230505202856862.png" alt="image-20230505202856862" width="25%" hight="25%"  style="zoom:50%;" />


**Datasets:** Stanford Dogs, Stanford Cars, CUB 200 Birds, and Flowers as benchmark datasets. Pascal VOC 2012 and COCO 2017, COCO Stuff-27, and COCO Stuff-3.

**Experimental Contents:** Unsupervised object discovery, unsupervised semantic segmentation.

### Object-Centric Slot Diffusion-2023 mar 20

https://arxiv.org/pdf/2303.10834.pdf

**Background:** Making object-centric learning applicable to complex natural scenes remains a major challenge.

**Introduction:** A new object-centric learning model, LSD, is proposed. 1⃣️From the perspective of object-centric learning, it replaces the traditional slot decoder with a latent **diffusion model** conditioned on object slots. 2⃣️From the perspective of the diffusion model, it is the first unsupervised compositional conditional diffusion model that does not require supervised annotations like text descriptions to learn compositions.

**Method: Encoder (consistent with traditional, obtains slot representations), Decoder using diffusion modeling (reconstructing images)**

<img src="https://github.com/Cmy61/Awesome-Object-Centric-Learning/blob/main/image/image-20230423192405624.png" alt="image-20230423192405624" width="70%" hight="70%"  style="zoom:50%;" />

* Encoder: Encodes the image into slot representations (slot attention).

* Decoder: Used to reconstruct the image given the slot representations S. The diffusion steps are used to denoise the latent variable z0 and generate a clearer reconstruction representation.

  * VQGAN: Transforms into a low-dimensional representation (z0) that can be decoded into a reconstructed image using the VQGAN decoder.
  * LSD Decoder: Utilizes diffusion modeling to reconstruct the latent representation z0 from VQGAN based on the slot representations S. (The decoder in LSD is conditioned on the slot representations S for conditional generation.)
    * Denoising Network: CNN LAYER + Slot-Conditioned Transformer.
      **Image -> Slot Representations -> VQGAN (dimensionality reduction to z0) -> LSD Decoder (denoising to generate latent variable z0) -> VQGAN Decoder (reconstruct into image)**


Applications: Generating synthetic images and allowing editing of existing images by directly modifying the slot representations.

Generating Synthetic Images:

**Image -> Slot Representations -> Clustering into K libraries -> Selecting K slots from the libraries and stacking them to form Scompose -> LSD Decoder (denoising to generate latent variable z0) -> VQGAN Decoder (reconstruct into image)**

Allowing Editing of Existing Images by Directly Modifying Slot Representations:

**Datasets:** FFHQ dataset, CLEVER (overfitting), CLEVR, CLEVRTex, MOVi-C, MOVi-E. Simple image effects are not as good as complex image effects.


### BRIDGING THE GAP TO REAL-WORLD OBJECTCENTRIC LEARNING-2023 mar 6

https://arxiv.org/pdf/2209.14860.pdf

**Background:** Current methods are limited to simulated data or require additional information in the form of motion or depth to successfully discover objects.

**Introduction:** Instead of relying on auxiliary external signals, this approach achieves object-centric representation in a fully **unsupervised** manner by using **reconstruction features as training signals** (introducing additional inductive bias by reconstructing highly homogeneous features within objects). These features can be easily obtained using recent self-supervised learning techniques such as DINO. **dinosaur = slot attention + DINO**

**Method:** Design similar to autoencoder: Module 1 extracts features from the input data (encoder); Module 2 groups them into a set of latent vectors representing slots; Module 3 (decoder) reconstructs some target signals from the latent vectors.

<img src="https://github.com/Cmy61/Awesome-Object-Centric-Learning/blob/main/image/image-20230508211006858.png" alt="image-20230508211006858" width="70%" hight="70%"  style="zoom:50%;" />

The key difference of this approach from other methods is that the task of the decoder is to reconstruct features from self-supervised pretraining, rather than reconstructing the original input.

* Feature Reconstruction as Training Signal: Previously, the task (at least initially) strongly focused on low-level image features such as color statistics. This quickly reduces the reconstruction error, but the generated model does not discover objects beyond the dataset, where objects are mainly determined by different object colors. Minimize the following loss to reconstruct self-supervised features:
* <img src="https://github.com/Cmy61/Awesome-Object-Centric-Learning/blob/main/image/image-20230508211924892.png" alt="image-20230508211924892" width="30%" hight="30%"  style="zoom:50%;" />
* Object-Centric Learning Architecture in the Real World:
  * Encoder: VIK encoder is used to learn feature representations.
  * Slot Attention: We primarily use the original Slot Attention formulation for grouping the set of features into slot vectors, with certain modifications: ① No position encoding is added to the ViT features before Slot Attention. ② A small single hidden layer MLP is added, which transforms each encoder feature before Slot Attention. (Enhances representation learning capability)
  * Feature Decoding: **We apply feature reconstruction instead of image reconstruction as the training objective**. Initially, an **MLP decoder** is used (the final reconstruction result y∈RN×Dfeat is obtained by weighted summing the reconstruction results of all slots), and later a **transformer decoder** is used (the Transformer decoder jointly reconstructs the feature y for all timesteps in an autoregressive manner, maintaining global consistency throughout the reconstruction process).

**Datasets:** MOVi dataset, MOVi-C and MOVi-E variants (with multiple objects), PASCAL VOC 2012 (real dataset with a large object), MS COCO 2017 (real dataset with several tens of objects) + KITTI driving dataset.


### InstMove: Instance Motion for Object-centric Video Segmentation-2023 mar

https://arxiv.org/pdf/2303.08132.pdf

**Task:** Segmentation and tracking of object instances in a given video. The goal is to design a flexible and efficient motion prediction module that can be easily integrated into any existing methods.

**Background:** State-of-the-art video segmentation methods are **sensitive to occlusion and fast motion**, making them susceptible to interference from these factors. Common methods **heavily rely on appearance changes**, which makes it difficult to handle multiple object instances with similar appearances, resulting in poor performance in complex scenes.

**Introduction:** InstMove represents instance motion for object-centric video segmentation. It primarily relies on instance-level motion information that is unaffected by image feature embeddings and has a physically interpretable nature, making it more accurate and robust for occluded and fast-moving objects. It learns dynamic models using a memory network to **predict** the position and shape of objects in the next frame.

**Method:** We utilize an instance mask to indicate the location and shape of the target object and employ an RNN-based module and a memory network to extract motion features from previous masks, store and retrieve dynamic information, and predict the shape information for the next frame based on motion clues.

<img src="https://github.com/Cmy61/Awesome-Object-Centric-Learning/blob/main/image/image-20230510151154704.png" alt="image-20230510151154704" width="70%" hight="70%"  style="zoom:50%;" />
* Memory Network
  * Step 1: **Current instance mask --> Extract current motion pattern z --> Find the most similar motion pattern ^z --> Predict the next state of z based on ^z** (Specifically, the encoder qφ extracts the motion pattern zkt−n:t=qφ(mkt−n:t) given the input. The memory network M stores representative motion patterns vi. When given the input motion pattern zkt−n:t, the attention weight vector wkt−n:t is computed to determine the most matching motion pattern ^z, which represents the motion pattern most similar to the current input zkt−n:t. By utilizing the stored representative motion pattern ^z in the memory bank, it assists in predicting the next state of z.) Train qφ and M.
  * Step 2: **Previous instance mask --> Extract motion pattern z --> Find the most similar motion pattern ^z --> Predict the next state of z based on ^z** (Specifically, the estimated mask mkt−n:t−1 from a target video segmentation method is used as input to predict the mask mkt. Using another encoder pθ, the motion pattern zkt−n:t−1=pθ(mkt−n:t−1) is extracted, converting the previous input mask sequence mkt−n:t−1 into the corresponding motion pattern zkt−n:t−1. Then, using the previously mentioned similar approach, the memory bank is accessed to match zkt−n:t−1 with the learned motion patterns and retrieve the corresponding motion pattern ^zkt−n:t−1.) Train pθ.
* Memory Network Motion Prediction: With the help of the memory network, a RNN-based network is used to predict the target frame mkt.
  * **Input motion mask --> Mask encoder (extract mask features fk) --> Combination of fk and ^z --> Mask decoder (predict target mask mtk)**
* Training:
  * Step 1: We train the encoder qφ and the memory bank M parameters using the input mkt−n:t. Then, ^zk(·) = ^zk(t−n:t) is used to predict the target mask mkt.
  * Step 2: We freeze the parameters of M and feed mkt−n:t−1 to the encoder pθ. We only train the encoder pθ in this step and use ^zk(·) = ^zk(t−n:t−1) for prediction.

**Datasets:** OVIS dataset, YouTubeVIS-Long dataset


### Shepherding Slots to Objects: Towards Stable and Robust Object-Centric Learning-2023

https://arxiv.org/pdf/2303.17842.pdf

**Task:** Object discovery for single image using OCL (Object-Contextual Representations) task.

**Background:** Existing models for single-view images suffer from bleeding issues, where slots capture different objects or objects tangled with the background. This is detrimental for OCL.

**Introduction:** The paper proposes a new OCL framework called SLASH (SLot Attention via SHepherding) to address the bleeding issue by guiding slots to successfully capture objects from random initialization. The key idea is to add two modules, namely **Attention Refining Kernel (ARK)** and **Intermediate Point Predictor and Encoder (IPPE)**, to the slot attention module.

* ARK is a single-channel single-layer convolutional kernel designed to prevent slots from focusing on noisy backgrounds.
* IPPE serves as a guidance to drive the slots towards the correct positions and incorporates positional information into the slots. However, this requires a weak semi-supervised approach.

**Method:** **Add two modules to the slot attention to address the bleeding issue.**

<img src="https://github.com/Cmy61/Awesome-Object-Centric-Learning/blob/main/image/image-20230510212831873.png" alt="image-20230510212831873" width="70%" hight="70%"  style="zoom:50%;" />

In this work, the model needs to provide guidance to the slots on what to attend and what not to attend. **ARK** protects and stabilizes the slots from background noise by reducing noise and solidifying class-object patterns in the attention maps between slots and pixels. **IPPE** guides the slots towards regions where objects might exist by providing positional cues to the slots.

* Slot Attention: The model adds ARK and IPPE modules to the slot attention model.
* ARK: It aims to prevent slots from being distracted by background noise by refining the attention maps between slots and visual features. The specific approach involves introducing inductive biases based on object local density to address this issue. The inductive biases of local density assume that the density of attention values should be higher near the objects and lower outside the objects (denser attention values closer to objects).
* <img src="https://github.com/Cmy61/Awesome-Object-Centric-Learning/blob/main/image/image-20230510213822596.png" alt="image-20230510213822596" width="50%" hight="50%"  style="zoom:50%;" />

(As shown in the upper part, the attention maps generated by Slot Attention exhibit salt-and-pepper noise around the objects.)

* IPPE: It aims to expedite learning "where objects exist." To enable IPPE to understand the position of objects, external supervision related to object positions is introduced (weak semi-supervision).

  * Point Predictor: A 3-layer MLP that predicts the 2D point coordinates of objects in the slots.
  * Point Encoder: A 3-layer MLP that encodes the point coordinates into a Dslot-dimensional vector, which is added to the original slots.

  <img src="https://github.com/Cmy61/Awesome-Object-Centric-Learning/blob/main/image/image-20230511200050337.png" alt="image-20230511200050337"  width="30%" hight="30%"  style="zoom:50%;" />

**Dataset:** CLEVRTEX+PTR +CLEVR6+MOVi-C
