# Official implementation of "A Unified Framework for Multimodal, Multi-Part Human Motion Synthesis"


### [Arxiv](https://arxiv.org/abs/2311.16471) [Project Website](https://zixiangzhou916.github.io/UDE-2/)

---

![plot](./assets/teaser.png)

## Abstract
#### The field has made significant progress in synthesizing realistic human motion driven by various modalities. Yet, the need for different methods to animate various body parts according to different control signals limits the scalability of these techniques in practical scenarios. In this paper, we introduce a cohesive and scalable approach that consolidates multimodal (text, music, speech) and multi-part (hand, torso) human motion generation. Our methodology unfolds in several steps: We begin by quantizing the motions of diverse body parts into separate codebooks tailored to their respective domains. Next, we harness the robust capabilities of pre-trained models to transcode multimodal signals into a shared latent space. We then translate these signals into discrete motion tokens by iteratively predicting subsequent tokens to form a complete sequence. Finally, we reconstruct the continuous actual motion from this tokenized sequence. Our method frames the multimodal motion generation challenge as a token prediction task, drawing from specialized codebooks based on the modality of the control signal. This approach is inherently scalable, allowing for the easy integration of new modalities. Extensive experiments demonstrated the effectiveness of our design, emphasizing its potential for broad application.

## Overview
![plot](assets/pipeline.png)
#### Hierarchical torso VQ-VAE:
- Encode the relative trajectory instead of global trajectory.
- Decode the discrete tokens to local poses(1st stage).
- Estimate sub-optimal global poses.
- Refine to obtain the final global poses(2nd stage).
- Introduce weights re-initialization to improve tokens activation rate.

#### Multimodal Multi-Part Motion Generation:
- Use large-scale pretrained models as encoders: 1) CLIP text encoder, 2) MTR, 3) HuBERT.
- Use encoder-decoder architecture to transform multimodal condition to motion tokens.
- Accept auxiliary condition as learnable embeddings.
- Introduce semantic enhancement module to align semantic between condition and motion.
- Introduce semantic-aware sampling to improve condition consistency while maintaining synthesis diversity.


## Code
#### Comming soon