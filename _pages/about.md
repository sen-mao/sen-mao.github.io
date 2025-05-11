---
permalink: /
title: ""
excerpt: ""
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

{% if site.google_scholar_stats_use_cdn %}
{% assign gsDataBaseUrl = "https://cdn.jsdelivr.net/gh/" | append: site.repository | append: "@" %}
{% else %}
{% assign gsDataBaseUrl = "https://raw.githubusercontent.com/" | append: site.repository | append: "/" %}
{% endif %}
{% assign url = gsDataBaseUrl | append: "google-scholar-stats/gs_data_shieldsio.json" %}

<span class='anchor' id='about-me'></span>

# Hi, there 👋

I am currently a Ph.D student in Nankai University from August 2022, supervised by Prof. <a href='https://scholar.google.com/citations?hl=en&user=6CsB8k0AAAAJ'>Yaxing Wang</a>. I obtained my master’s degree in Computer Technology from the College of Computer Science, Nankai University.

My research interests include **Generative Models**, **Image Generation**, and **Image-to-image Translation**. 

I’m currently conducting some research in image editing and efficient inference, including:

🎨 Image editing based on Generative Models (GANs and Diffusion Models).

🚀 The acceleration of inferecne by training-free or data-free distillation.





# 🔥 News
- <span style="color: black; font-weight: bold;">[2025.02]</span> &nbsp;🥳🥳 Two papers (including one co-authored paper MaskUNet) accepted by <span style="color: red; font-weight: bold;">CVPR2025</span>. <br> <span style="color: black; font-weight: bold;">✨One-Way Ticket : Time-Independent Unified Encoder for Distilling Text-to-Image Diffusion Model</span>. See <a href='https://github.com/sen-mao/Loopfree'>Github</a>.
- <span style="color: black; font-weight: bold;">[2025.01]</span> &nbsp;🥳🥳 Two papers (including one co-authored paper <a href='https://arxiv.org/pdf/2501.13554'>1Prompt1Story</a>) accepted by <span style="color: red; font-weight: bold;">ICLR2025</span>. <br> <span style="color: black; font-weight: bold;">✨InterLCM: Low-Quality Images as Intermediate States of Latent Consistency Models for Effective Blind Face Restoration</span>. See <a href='https://arxiv.org/pdf/2502.02215'>paper</a> and <a href='https://sen-mao.github.io/InterLCM-Page/'>Project Page</a>.
- <span style="color: black; font-weight: bold;">[2024.09]</span> &nbsp;🥳🥳 <span style="color: black; font-weight: bold;">StyleDiffusion: Prompt-Embedding Inversion for Text-Based Editing</span> accepted by <span style="color: red; font-weight: bold;">CVMJ2024</span>. See <a href='https://arxiv.org/pdf/2303.15649'>paper</a> and <a href='https://github.com/sen-mao/StyleDiffusion'>code</a>.
- <span style="color: black; font-weight: bold;">[2024.09]</span> &nbsp;🥳🥳 <span style="color: black; font-weight: bold;">Faster Diffusion: Rethinking the Role of the Encoder for Diffusion Model Inference</span> accepted by <span style="color: red; font-weight: bold;">NeurIPS2024</span>. See <a href='https://arxiv.org/pdf/2312.09608'>paper</a> and <a href='https://sen-mao.github.io/FasterDiffusion/'>Project Page</a>.
- <span style="color: black; font-weight: bold;">[2024.01]</span> &nbsp;🥳🥳 <span style="color: black; font-weight: bold;">Get What You Want, Not What You Don't: Image Content Suppression for Text-to-Image Diffusion Models</span> accepted by <span style="color: red; font-weight: bold;">ICLR2024</span>. See <a href='https://arxiv.org/abs/2402.05375'>paper</a> and <a href='https://github.com/sen-mao/SuppressEOT'>code</a>.
- <span style="color: black; font-weight: bold;">[2023.12]</span> &nbsp;🎉🎉 <span style="color: red; font-weight: bold;">New work</span> <span style="color: black; font-weight: bold;">Faster Diffusion: Rethinking the Role of UNet Encoder in Diffusion Models</span>. See <a href='https://arxiv.org/abs/2312.09608'>paper</a> and <a href='https://github.com/hutaiHang/Faster-Diffusion'>code</a>.
- <span style="color: black; font-weight: bold;">[2023.02]</span> &nbsp;🥳🥳 <span style="color: black; font-weight: bold;">3D-Aware Multi-Class Image-to-Image Translation with NeRFs</span> accepted by <span style="color: red; font-weight: bold;">CVPR2023</span>. See <a href='https://arxiv.org/abs/2303.15012'>paper</a> and <a href='https://github.com/sen-mao/3di2i-translation'>code</a>.
- <span style="color: black; font-weight: bold;">[2020.12]</span> &nbsp;🥳🥳 <span style="color: black; font-weight: bold;">Low-rank Constrained Super-Resolution for Mixed-Resolution Multiview Video</span> accepted by <span style="color: red; font-weight: bold;">TIP2020</span>. See <a href='https://ieeexplore.ieee.org/abstract/document/9286862'>paper</a> and <a href='https://drive.google.com/file/d/1spFEH6H1jMWZB2vqhU-PQ8ruhJ-VOHf-/view?usp=sharing'>code</a>.


# 📝 Publications 

[//]: # (Loopfree)
<div class='paper-box'><div class='paper-box-image'><div><div class="badge">CVPR 2025</div><img src='images/papers/Loopfree.jpg' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[One-Way Ticket : Time-Independent Unified Encoder for Distilling Text-to-Image Diffusion Model](https://github.com/sen-mao/Loopfree)

<span style="color: #0000FF;"><strong>Senmao Li</strong></span>, Lei Wang, Kai Wang, Tao Liu, Jiehang Xie, Joost van de Weijier, Fahad Shahbaz Khan, Shiqi Yang, Yaxing Wang*, Jian Yang

- We introduce the first Time-independent Unified Encoder (TiUE) architecture, which is a loop-free distillation approach and eliminates the need for iterative noisy latent processing while maintaining high sampling fidelity with a time cost comparable to previous one-step methods.

<div style="display: inline">
        <a href="https://drive.google.com/file/d/1CenAPN9qPvBhDCc0Za6Fx6yXGXZCMpst/view?usp=sharing"> [中译版]</a>
        <a href="https://github.com/sen-mao/Loopfree"> [code]</a>
        <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" >[abstract]</a>
        <div class="abstract"  style="overflow: hidden; display: none;">  
            <p> Text-to-Image (T2I) diffusion models have made remarkable advancements in generative modeling; however, 
                they face a trade-off between inference speed and image quality, posing challenges for efficient deployment.
                Existing distilled T2I models can generate high-fidelity images with fewer sampling steps, but often struggle with diversity and quality, especially in one-step models. 
                From our analysis, we observe redundant computations in the UNet encoders. 
                Our findings suggest that, for T2I diffusion models, decoders are more adept at capturing richer and more explicit semantic information, 
                while encoders can be effectively shared across decoders from diverse time steps.
                Based on these observations, we introduce the first Time-independent Unified Encoder (TiUE) for the student model UNet architecture,
                which is a loop-free image generation approach for distilling T2I diffusion models. Using a one-pass scheme,
                TiUE shares encoder features across multiple decoder time steps, enabling parallel sampling and significantly reducing inference time complexity.
                In addition, we incorporate a KL divergence term to regularize noise prediction, which enhances the perceptual realism and diversity of the generated images.
                Experimental results demonstrate that TiUE outperforms state-of-the-art methods, including LCM, SD-Turbo, and SwiftBrushv2, producing more diverse and realistic results while maintaining the computational efficiency.
            </p>
        </div>
</div>

</div>
</div>

[//]: # (InterLCM)
<div class='paper-box'><div class='paper-box-image'><div><div class="badge">ICLR 2025</div><img src='images/papers/InterLCM.jpg' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[InterLCM: Low-Quality Images as Intermediate States of Latent Consistency Models for Effective Blind Face Restoration](https://arxiv.org/abs/2502.02215)

<span style="color: #0000FF;"><strong>Senmao Li</strong></span>, Kai Wang*, Joost van de Weijier, Fahad Shahbaz Khan, Chun-Le Guo, Shiqi Yang, Yaxing Wang, Jian Yang, Ming-Ming Cheng

- By considering the low-quality image as the intermediate state of LCM models, we can effectively maintain better semantic consistency in face restorations.
- Our method InterLCM has additional advantages: few-step sampling with much faster speed and integrating our framework with commonly used perceptual loss and adversarial loss in face restoration. 

<div style="display: inline">
        <a href="https://arxiv.org/abs/2502.02215"> [paper]</a>
        <a href="https://github.com/sen-mao/InterLCM"> [code]</a>
        <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" >[abstract]</a>
        <div class="abstract"  style="overflow: hidden; display: none;">  
            <p> Diffusion priors have been used for blind face restoration (BFR) by fine-tuning diffusion models (DMs) on restoration datasets to recover low-quality images. However, the naive application of DMs presents several key limitations. 
                (i) The diffusion prior has inferior semantic consistency (e.g., ID, structure and color.),  increasing the difficulty of optimizing the BFR model;
                (ii) reliance on hundreds of denoising iterations, preventing the effective cooperation with perceptual losses, which is crucial for faithful restoration.
                Observing that the latent consistency model (LCM) learns consistency noise-to-data mappings on the ODE-trajectory and therefore shows more semantic consistency in the subject identity, structural information and color preservation, 
                we propose InterLCM to leverage the LCM for its superior semantic consistency and efficiency to counter the above issues. 
                Treating low-quality images as the intermediate state of LCM, InterLCM achieves a balance between fidelity and quality by starting from earlier LCM steps. 
                LCM also allows the integration of perceptual loss during training, leading to improved restoration quality, particularly in real-world scenarios.
                To mitigate structural and semantic uncertainties, InterLCM incorporates a Visual Module to extract visual features and a Spatial Encoder to capture spatial details, enhancing the fidelity of restored images.
                Extensive experiments demonstrate that InterLCM outperforms existing approaches in both synthetic and real-world datasets while also achieving faster inference speed. 
            </p>
        </div>
</div>

</div>
</div>

[//]: # (FasterDiffusion)
<div class='paper-box'><div class='paper-box-image'><div><div class="badge">NeurIPS 2024</div><img src='images/papers/FasterDiffusion.jpg' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[Faster Diffusion: Rethinking the Role of the Encoder for Diffusion Model Inference](https://arxiv.org/abs/2312.09608)

<span style="color: #0000FF;"><strong>Senmao Li</strong></span>, Taihang Hu, Joost van de Weijier, Fahad Shahbaz Khan, Linxuan Li, Shiqi Yang, Yaxing Wang*, Ming-Ming Cheng, Jian Yang

- A thorough empirical study of the features of the UNet in the diffusion model showing that encoder features vary minimally (whereas decoder feature vary significantly)
- An encoder propagation scheme to accelerate the diffusion sampling without requiring any training or fine-tuning technique
- ~1.8x acceleration for stable diffusion, 50 DDIM steps, ~1.8x acceleration for stable diffusion, 20 Dpm-solver++ steps, and ~1.3x acceleration for DeepFloyd-IF

<div style="display: inline">
        <a href="https://arxiv.org/abs/2312.09608"> [paper]</a>|<a href="https://drive.google.com/file/d/1NEgrFM3kxLoPs2dWubAbYuqqFL4EDoOx/view?usp=sharing">[中译版]</a>
        <a href="https://sen-mao.github.io/FasterDiffusion/"> [code]</a>
        <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" >[abstract]</a>
        <div class="abstract"  style="overflow: hidden; display: none;">  
            <p> One of the key components within diffusion models is the UNet for noise prediction. While several works have explored basic properties of the UNet decoder, its encoder largely remains unexplored. In this work, we conduct the first comprehensive study of the UNet encoder. We empirically analyze the encoder features and provide insights to important questions regarding their changes at the inference process. In particular, we find that encoder features change gently, whereas the decoder features exhibit substantial variations across different time-steps. This finding inspired us to omit the encoder at certain adjacent time-steps and reuse cyclically the encoder features in the previous time-steps for the decoder. Further based on this observation, we introduce a simple yet effective encoder propagation scheme to accelerate the diffusion sampling for a diverse set of tasks. By benefiting from our propagation scheme, we are able to perform in parallel the decoder at certain adjacent time-steps. Additionally, we introduce a prior noise injection method to improve the texture details in the generated image. Besides the standard text-to-image task, we also validate our approach on other tasks: text-to-video, personalized generation and reference-guided generation. Without utilizing any knowledge distillation technique, our approach accelerates both the Stable Diffusion (SD) and the DeepFloyd-IF models sampling by 41% and 24% respectively, while maintaining high-quality generation performance. </p>
        </div>
</div>

</div>
</div>


[//]: # (SuppressEOT)
<div class='paper-box'><div class='paper-box-image'><div><div class="badge">ICLR 2024</div><img src='images/papers/SuppressEOT.jpg' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[Get What You Want, Not What You Don't: Image Content Suppression for Text-to-Image Diffusion Model](https://arxiv.org/pdf/2402.05375.pdf)

<span style="color: #0000FF;"><strong>Senmao Li</strong></span>, Joost van de Weijer, Taihang Hu, Fahad Shahbaz Khan, Qibin Hou, Yaxing Wang*, Jian Yang

- The [EOT] embeddings contain significant, redundant and duplicated semantic information of the whole input prompt.
- We propose soft-weighted regularization (SWR) to eliminate the negative target information from the [EOT] embeddings.
- We propose inference-time text embedding optimization (ITO).

<div style="display: inline">
        <a href="https://arxiv.org/abs/2402.05375"> [paper]</a>|<a href="https://drive.google.com/file/d/1CXDAO7WzmZloF2TTHQibRPIKYjKliMoc/view?usp=sharing">[中译版]</a>
        <a href="https://github.com/sen-mao/SuppressEOT"> [code]</a>
        <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" >[abstract]</a>
        <div class="abstract"  style="overflow: hidden; display: none;">  
            <p> The success of recent text-to-image diffusion models is largely due to their capacity to be guided by a complex text prompt, which enables users to precisely describe the desired content. However, these models struggle to effectively suppress the generation of undesired content, which is explicitly requested to be omitted from the generated image in the prompt. In this paper, we analyze how to manipulate the text embeddings and remove unwanted content from them. We introduce two approaches, which we refer to as **soft-weighted regularization** and **inference-time text embedding optimization**. The first regularizes the text embedding matrix and effectively suppresses the undesired content. The second method aims to further suppress the unwanted content generation of the prompt, and encourages the generation of desired content. We evaluate our method quantitatively and qualitatively on extensive experiments, validating its effectiveness. Furthermore, our method is generalizability to both the pixel-space diffusion models (i.e. DeepFloyd-IF) and the latent-space diffusion models (i.e. Stable Diffusion).</p>
        </div>
        <a href="https://zhuanlan.zhihu.com/p/700936543"> [中文解读]</a>
</div>

</div>
</div>

[//]: # (StyleDiffusion)
<div class='paper-box'><div class='paper-box-image'><div><div class="badge">CVMJ 2024</div><img src='images/papers/StyleDiffusion.jpg' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[StyleDiffusion: Prompt-Embedding Inversion for Text-Based Editing](https://arxiv.org/abs/2303.15649)

<span style="color: #0000FF;"><strong>Senmao Li</strong></span>, Joost van de Weijer, Taihang Hu, Fahad Shahbaz Khan, Qibin Hou, Yaxing Wang*, Jian Yang

- Only optimizing the input of the value linear network in the cross-attention layers is sufficiently powerful to reconstruct a real image
- Attention regularization to preserve the object-like attention maps after reconstruction and editing, enabling us to obtain accurate style editing without invoking significant structural changes

<div style="display: inline">
        <a href="https://arxiv.org/abs/2303.15649"> [paper]</a>|<a href="https://drive.google.com/file/d/1k0BEPCe8lf24Ejsc0ir-MAashSRTWpXJ/view?usp=sharing">[中译版]</a>
        <a href="https://github.com/sen-mao/StyleDiffusion"> [code]</a>
        <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" >[abstract]</a>
        <div class="abstract"  style="overflow: hidden; display: none;">  
            <p> A significant research effort is focused on exploiting the amazing capacities of pretrained diffusion models for the editing of images. They either finetune the model, or invert the image in the latent space of the pretrained model. However, they suffer from two problems: (1) Unsatisfying results for selected regions, and unexpected changes in nonselected regions. (2) They require careful text prompt editing where the prompt should include all visual objects in the input image. To address this, we propose two improvements: (1) Only optimizing the input of the value linear network in the cross-attention layers, is sufficiently powerful to reconstruct a real image. (2) We propose attention regularization to preserve the object-like attention maps after editing, enabling us to obtain accurate style editing without invoking significant structural changes. We further improve the editing technique which is used for the unconditional branch of classifier-free guidance, as well as the conditional one as used by P2P. Extensive experimental prompt-editing results on a variety of images, demonstrate qualitatively and quantitatively that our method has superior editing capabilities than existing and concurrent works.</p>
        </div>
        <a href="https://zhuanlan.zhihu.com/p/12454546331"> [中文解读]</a>
</div>

</div>
</div>


[//]: # (3DI2I)
<div class='paper-box'><div class='paper-box-image'><div><div class="badge">CVPR 2023</div><img src='images/papers/3DI2I.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[3D-Aware Multi-Class Image-to-Image Translation with NeRFs](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_3D-Aware_Multi-Class_Image-to-Image_Translation_With_NeRFs_CVPR_2023_paper.pdf)

<span style="color: #0000FF;"><strong>Senmao Li</strong></span>, Joost van de Weijer, Yaxing Wang*, Fahad Shahbaz Khan, Meiqin Liu, Jian Yang

- The first to explore 3D-aware multi-class I2I translation
- Decouple 3D-aware I2I translation into two steps

<div style="display: inline">
        <a href="https://arxiv.org/abs/2303.15012"> [paper]</a>|<a href="https://drive.google.com/file/d/1QzaTAieNEluN6FdSHTZ09Q7SdM75yWuF/view?usp=sharing">[中译版]</a>
        <a href="https://github.com/sen-mao/3di2i-translation"> [code]</a>
        <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" >[abstract]</a>
        <div class="abstract"  style="overflow: hidden; display: none;">  
            <p> Recent advances in 3D-aware generative models (3D-aware GANs) combined with Neural Radiance Fields (NeRF) have achieved impressive results for novel view synthesis. However no prior works investigate 3D-aware GANs for 3D consistent multi-class image-to-image (3D-aware I2I) translation. Naively using 2D-I2I translation methods suffers from unrealistic shape/identity change. To perform 3D-aware multi-class I2I translation, we decouple this learning process into a multi-class 3D-aware GAN step and a 3D-aware I2I translation step. In the first step, we propose two novel techniques: a new conditional architecture and a effective training strategy. In the second step, based on the well-trained multi-class 3D-aware GAN architecture that preserves view-consistency, we construct a 3D-aware I2I translation system. To further reduce the view-consistency problems, we propose several new techniques, including a U-net-like adaptor network design, a hierarchical representation constrain and a relative regularization loss. In extensive experiments on two datasets, quantitative and qualitative results demonstrate that we successfully perform 3D-aware I2I translation with multi-view consistency.</p>
        </div>
</div>

</div>
</div>


<ul>
  <li>
    <img src='https://img.shields.io/github/stars/sen-mao/InterLCM' alt="sym" height="100%">
    <a href="https://arxiv.org/abs/2502.02215"> InterLCM: Low-Quality Images as Intermediate States of Latent Consistency Models for Effective Blind Face Restoration </a>. <span style="color: #0000FF;"><strong>Senmao Li</strong></span>, Kai Wang, Joost van de Weijier, Fahad Shahbaz Khan, Chun-Le Guo, Shiqi Yang, Yaxing Wang, Jian Yang, Ming-Ming Cheng. <strong>ICLR2025</strong>. 
    <div style="display: inline">
        <a href="https://arxiv.org/abs/2502.02215"> [paper]</a>
        <a href="https://sen-mao.github.io/InterLCM-Page/"> [Project Page]</a>
        <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" >[abstract]</a>
        <div class="abstract"  style="overflow: hidden; display: none;">  
            <p> Diffusion priors have been used for blind face restoration (BFR) by fine-tuning diffusion models (DMs) on restoration datasets to recover low-quality images. However, the naive application of DMs presents several key limitations. 
                (i) The diffusion prior has inferior semantic consistency (e.g., ID, structure and color.),  increasing the difficulty of optimizing the BFR model;
                (ii) reliance on hundreds of denoising iterations, preventing the effective cooperation with perceptual losses, which is crucial for faithful restoration.
                Observing that the latent consistency model (LCM) learns consistency noise-to-data mappings on the ODE-trajectory and therefore shows more semantic consistency in the subject identity, structural information and color preservation, 
                we propose InterLCM to leverage the LCM for its superior semantic consistency and efficiency to counter the above issues. 
                Treating low-quality images as the intermediate state of LCM, InterLCM achieves a balance between fidelity and quality by starting from earlier LCM steps. 
                LCM also allows the integration of perceptual loss during training, leading to improved restoration quality, particularly in real-world scenarios.
                To mitigate structural and semantic uncertainties, InterLCM incorporates a Visual Module to extract visual features and a Spatial Encoder to capture spatial details, enhancing the fidelity of restored images.
                Extensive experiments demonstrate that InterLCM outperforms existing approaches in both synthetic and real-world datasets while also achieving faster inference speed. 
            </p>
        </div>
    </div>
  </li>

  <li>
    <img src='https://img.shields.io/github/stars/hutaiHang/Faster-Diffusion' alt="sym" height="100%">
    <a href="https://arxiv.org/abs/2312.09608"> Faster Diffusion: Rethinking the Role of the Encoder for Diffusion Model Inference </a>. <span style="color: #0000FF;"><strong>Senmao Li</strong></span>, Taihang Hu, Joost van de Weijier, Fahad Shahbaz Khan, Linxuan Li, Shiqi Yang, Yaxing Wang, Ming-Ming Cheng, Jian Yang. <strong>NeurIPS2024</strong>. 
    <div style="display: inline">
        <a href="https://arxiv.org/abs/2312.09608"> [paper]</a>
        <a href="https://sen-mao.github.io/FasterDiffusion/"> [Project Page]</a>
        <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" >[abstract]</a>
        <div class="abstract"  style="overflow: hidden; display: none;">  
            <p> One of the key components within diffusion models is the UNet for noise prediction. While several works have explored basic properties of the UNet decoder, its encoder largely remains unexplored. In this work, we conduct the first comprehensive study of the UNet encoder. We empirically analyze the encoder features and provide insights to important questions regarding their changes at the inference process. In particular, we find that encoder features change gently, whereas the decoder features exhibit substantial variations across different time-steps. This finding inspired us to omit the encoder at certain adjacent time-steps and reuse cyclically the encoder features in the previous time-steps for the decoder. Further based on this observation, we introduce a simple yet effective encoder propagation scheme to accelerate the diffusion sampling for a diverse set of tasks. By benefiting from our propagation scheme, we are able to perform in parallel the decoder at certain adjacent time-steps. Additionally, we introduce a prior noise injection method to improve the texture details in the generated image. Besides the standard text-to-image task, we also validate our approach on other tasks: text-to-video, personalized generation and reference-guided generation. Without utilizing any knowledge distillation technique, our approach accelerates both the Stable Diffusion (SD) and the DeepFloyd-IF models sampling by 41% and 24% respectively, while maintaining high-quality generation performance. </p>
        </div>
    </div>
  </li>


  <li>
    <img src='https://img.shields.io/github/stars/sen-mao/SuppressEOT' alt="sym" height="100%">
    <a href="https://arxiv.org/abs/2402.05375"> Get What You Want, Not What You Don't: Image Content Suppression for Text-to-Image Diffusion Models </a>. <span style="color: #0000FF;"><strong>Senmao Li</strong></span>, Joost van de Weijer, Taihang Hu, Fahad Shahbaz Khan, Qibin Hou, Yaxing Wang, Jian Yang. <strong>ICLR 2024</strong>. 
    <div style="display: inline">
        <a href="https://arxiv.org/abs/2402.05375"> [paper]</a>
        <a href="https://github.com/sen-mao/SuppressEOT"> [code]</a>
        <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" >[abstract]</a>
        <div class="abstract"  style="overflow: hidden; display: none;">  
            <p> The success of recent text-to-image diffusion models is largely due to their capacity to be guided by a complex text prompt, which enables users to precisely describe the desired content. However, these models struggle to effectively suppress the generation of undesired content, which is explicitly requested to be omitted from the generated image in the prompt. In this paper, we analyze how to manipulate the text embeddings and remove unwanted content from them. We introduce two approaches, which we refer to as **soft-weighted regularization** and **inference-time text embedding optimization**. The first regularizes the text embedding matrix and effectively suppresses the undesired content. The second method aims to further suppress the unwanted content generation of the prompt, and encourages the generation of desired content. We evaluate our method quantitatively and qualitatively on extensive experiments, validating its effectiveness. Furthermore, our method is generalizability to both the pixel-space diffusion models (i.e. DeepFloyd-IF) and the latent-space diffusion models (i.e. Stable Diffusion).</p>
        </div>
    </div>
  </li>



  <li>
    <img src='https://img.shields.io/github/stars/sen-mao/StyleDiffusion' alt="sym" height="100%">
    <a href="https://arxiv.org/abs/2312.09608"> StyleDiffusion: Prompt-Embedding Inversion for Text-Based Editing </a>. <span style="color: #0000FF;"><strong>Senmao Li</strong></span>, Joost van de Weijer, Taihang Hu, Fahad Shahbaz Khan, Qibin Hou, Yaxing Wang, Jian Yang. <strong>CVMJ 2024</strong>. 
    <div style="display: inline">
        <a href="https://arxiv.org/abs/2303.15649"> [paper]</a>
        <a href="https://github.com/sen-mao/StyleDiffusion"> [code]</a>
        <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" >[abstract]</a>
        <div class="abstract"  style="overflow: hidden; display: none;">  
            <p> A significant research effort is focused on exploiting the amazing capacities of pretrained diffusion models for the editing of images. They either finetune the model, or invert the image in the latent space of the pretrained model. However, they suffer from two problems: (1) Unsatisfying results for selected regions, and unexpected changes in nonselected regions. (2) They require careful text prompt editing where the prompt should include all visual objects in the input image. To address this, we propose two improvements: (1) Only optimizing the input of the value linear network in the cross-attention layers, is sufficiently powerful to reconstruct a real image. (2) We propose attention regularization to preserve the object-like attention maps after editing, enabling us to obtain accurate style editing without invoking significant structural changes. We further improve the editing technique which is used for the unconditional branch of classifier-free guidance, as well as the conditional one as used by P2P. Extensive experimental prompt-editing results on a variety of images, demonstrate qualitatively and quantitatively that our method has superior editing capabilities than existing and concurrent works.</p>
        </div>
    </div>
  </li>


  <li>
    <img src='https://img.shields.io/github/stars/sen-mao/3di2i-translation' alt="sym" height="100%">
    <a href="https://arxiv.org/abs/2303.15012"> 3D-Aware Multi-Class Image-to-Image Translation with NeRFs </a>. <span style="color: #0000FF;"><strong>Senmao Li</strong></span>, Joost van de Weijer, Yaxing Wang, Fahad Shahbaz Khan, Meiqin Liu, Jian Yang. <strong>CVPR 2023</strong>. 
    <div style="display: inline">
        <a href="https://arxiv.org/abs/2303.15012"> [paper]</a>
        <a href="https://github.com/sen-mao/3di2i-translation"> [code]</a>
        <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" >[abstract]</a>
        <div class="abstract"  style="overflow: hidden; display: none;">  
            <p> Recent advances in 3D-aware generative models (3D-aware GANs) combined with Neural Radiance Fields (NeRF) have achieved impressive results for novel view synthesis. However no prior works investigate 3D-aware GANs for 3D consistent multi-class image-to-image (3D-aware I2I) translation. Naively using 2D-I2I translation methods suffers from unrealistic shape/identity change. To perform 3D-aware multi-class I2I translation, we decouple this learning process into a multi-class 3D-aware GAN step and a 3D-aware I2I translation step. In the first step, we propose two novel techniques: a new conditional architecture and a effective training strategy. In the second step, based on the well-trained multi-class 3D-aware GAN architecture that preserves view-consistency, we construct a 3D-aware I2I translation system. To further reduce the view-consistency problems, we propose several new techniques, including a U-net-like adaptor network design, a hierarchical representation constrain and a relative regularization loss. In extensive experiments on two datasets, quantitative and qualitative results demonstrate that we successfully perform 3D-aware I2I translation with multi-view consistency.</p>
        </div>
    </div>
  </li>
</ul>

[//]: # (  ~~<li>)

[//]: # (    <img src='https://img.shields.io/github/stars/hutaiHang/Faster-Diffusion' alt="sym" height="100%">)

[//]: # (    <a href="https://arxiv.org/abs/2312.09608"> Faster Diffusion: Rethinking the Role of UNet Encoder in Diffusion Models </a>. <strong>Senmao Li</strong>, Taihang Hu, Fahad Khan, Linxuan Li, Shiqi Yang, Yaxing Wang, Ming-Ming Cheng, Jian Yang. <strong>arXiv</strong>. )

[//]: # (    <div style="display: inline">)

[//]: # (        <a href="https://arxiv.org/abs/2312.09608"> [paper]</a>)

[//]: # (        <a href="https://github.com/hutaiHang/Faster-Diffusion"> [code]</a>)

[//]: # (        <a class="fakelink" onclick="$&#40;this&#41;.siblings&#40;'.abstract'&#41;.slideToggle&#40;&#41;" >[abstract]</a>)

[//]: # (        <div class="abstract"  style="overflow: hidden; display: none;">  )

[//]: # (            <p> One of the key components within diffusion models is the UNet for noise prediction. While several works have explored basic properties of the UNet decoder, its encoder largely remains unexplored. In this work, we conduct the first comprehensive study of the UNet encoder. We empirically analyze the encoder features and provide insights to important questions regarding their changes at the inference process. In particular, we find that encoder features change gently, whereas the decoder features exhibit substantial variations across different time-steps. This finding inspired us to omit the encoder at certain adjacent time-steps and reuse cyclically the encoder features in the previous time-steps for the decoder. Further based on this observation, we introduce a simple yet effective encoder propagation scheme to accelerate the diffusion sampling for a diverse set of tasks. By benefiting from our propagation scheme, we are able to perform in parallel the decoder at certain adjacent time-steps. Additionally, we introduce a prior noise injection method to improve the texture details in the generated image. Besides the standard text-to-image task, we also validate our approach on other tasks: text-to-video, personalized generation and reference-guided generation. Without utilizing any knowledge distillation technique, our approach accelerates both the Stable Diffusion &#40;SD&#41; and the DeepFloyd-IF models sampling by 41% and 24% respectively, while maintaining high-quality generation performance. </p>)

[//]: # (        </div>)

[//]: # (    </div>)

[//]: # (  </li>~~)


[//]: # (# 🎖 Honors and Awards)

[//]: # (- *2021.10* Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus ornare aliquet ipsum, ac tempus justo dapibus sit amet. )

[//]: # (- *2021.09* Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus ornare aliquet ipsum, ac tempus justo dapibus sit amet. )

[//]: # ()
[//]: # (# 📖 Educations)

[//]: # (- *2019.06 - 2022.04 &#40;now&#41;*, Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus ornare aliquet ipsum, ac tempus justo dapibus sit amet. )

[//]: # (- *2015.09 - 2019.06*, Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus ornare aliquet ipsum, ac tempus justo dapibus sit amet. )

[//]: # (# 💬 Invited Talks)

[//]: # (- *2021.06*, Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus ornare aliquet ipsum, ac tempus justo dapibus sit amet. )

[//]: # (- *2021.03*, Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus ornare aliquet ipsum, ac tempus justo dapibus sit amet.  \| [\[video\]]&#40;https://github.com/&#41;)

# 📄 Academic Service
- *Conference Reviewer:* CVPR'25, [ICLR2025](https://iclr.cc/Conferences/2025/ProgramCommittee#all-reviewer), [NeurIPS202424](https://neurips.cc/Conferences/2024/ProgramCommittee#all-reviewers)



# 💻 Internships

- *2024.07 - 2024.09*, A research stay, supervised by Dr. [Joost van de Weijer](https://scholar.google.com/citations?user=Gsw2iUEAAAAJ) in [Computer Vision Center](https://www.cvc.uab.es/), Autonomous University of Barcelona, Spain.
