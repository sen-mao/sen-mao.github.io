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
- *2024.09*: &nbsp;🥳🥳 Our paper "StyleDiffusion: Prompt-Embedding Inversion for Text-Based Editing" accepted by CVMJ'24. See our <a href='https://arxiv.org/pdf/2303.15649'>paper</a> and <a href='https://github.com/sen-mao/StyleDiffusion'>code</a>.
- *2024.09*: &nbsp;🥳🥳 Our paper "Faster Diffusion: Rethinking the Role of the Encoder for Diffusion Model Inference" accepted by NeurIPS'24. See our <a href='https://arxiv.org/pdf/2312.09608'>paper</a> and <a href='https://sen-mao.github.io/FasterDiffusion/'>code</a>.
- *2024.01*: &nbsp;🥳🥳 Our paper "Get What You Want, Not What You Don't: Image Content Suppression for Text-to-Image Diffusion Models" accepted by ICLR'24. See our <a href='https://arxiv.org/abs/2402.05375'>paper</a> and <a href='https://github.com/sen-mao/SuppressEOT'>code</a>.
- *2023.12*: &nbsp;🎉🎉 Our new work, "FasterDiffusion: Rethinking the Role of UNet Encoder in Diffusion Models". See our <a href='https://arxiv.org/abs/2312.09608'>paper</a> and <a href='https://github.com/hutaiHang/Faster-Diffusion'>code</a>.
- *2023.02*: &nbsp;🥳🥳 Our paper "3D-Aware Multi-Class Image-to-Image Translation with NeRFs" accepted by CVPR'23. See our <a href='https://arxiv.org/abs/2303.15012'>paper</a> and <a href='https://github.com/sen-mao/3di2i-translation'>code</a>.
- *2020.12*: &nbsp;🥳🥳 Our paper "Low-rank Constrained Super-Resolution for Mixed-Resolution Multiview Video" accepted by TIP'20. See our <a href='https://ieeexplore.ieee.org/abstract/document/9286862'>paper</a> and <a href='https://drive.google.com/file/d/1spFEH6H1jMWZB2vqhU-PQ8ruhJ-VOHf-/view?usp=sharing'>code</a>.


# 📝 Publications 

[//]: # (FasterDiffusion)
<div class='paper-box'><div class='paper-box-image'><div><div class="badge">NeurIPS 2024</div><img src='images/papers/FasterDiffusion.jpg' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[Faster Diffusion: Rethinking the Role of the Encoder for Diffusion Model Inference](https://arxiv.org/abs/2312.09608)

**Senmao Li**, Taihang Hu, Joost van de Weijier, Fahad Khan, Linxuan Li, Shiqi Yang, Yaxing Wang, Ming-Ming Cheng, Jian Yang

- A thorough empirical study of the features of the UNet in the diffusion model showing that encoder features vary minimally (whereas decoder feature vary significantly)
- An encoder propagation scheme to accelerate the diffusion sampling without requiring any training or fine-tuning technique
- Our method can be combined with existing methods (like DDIM, and DPM-solver) to further accelerate diffusion model inference time
- ~1.8x acceleration for stable diffusion, 50 DDIM steps, ~1.8x acceleration for stable diffusion, 20 Dpm-solver++ steps, and ~1.3x acceleration for DeepFloyd-IF

<div style="display: inline">
        <a href="https://arxiv.org/abs/2312.09608"> [paper]</a>
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

**Senmao Li**, Joost van de Weijer, Taihang Hu, Fahad Shahbaz Khan, Qibin Hou, Yaxing Wang, Jian Yang

- The [EOT] embeddings contain significant, redundant and duplicated semantic information of the whole input prompt.
- We propose soft-weighted regularization (SWR) to eliminate the negative target information from the [EOT] embeddings.
- We propose inference-time text embedding optimization (ITO).

<div style="display: inline">
        <a href="https://arxiv.org/abs/2402.05375"> [paper]</a>
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

**Senmao Li**, Joost van de Weijer, Taihang Hu, Fahad Shahbaz Khan, Qibin Hou, Yaxing Wang, Jian Yang

- Only optimizing the input of the value linear network in the cross-attention layers is sufficiently powerful to reconstruct a real image
- Attention regularization to preserve the object-like attention maps after reconstruction and editing, enabling us to obtain accurate style editing without invoking significant structural changes

<div style="display: inline">
        <a href="https://arxiv.org/abs/2303.15649"> [paper]</a>
        <a href="https://github.com/sen-mao/StyleDiffusion"> [code]</a>
        <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" >[abstract]</a>
        <div class="abstract"  style="overflow: hidden; display: none;">  
            <p> A significant research effort is focused on exploiting the amazing capacities of pretrained diffusion models for the editing of images. They either finetune the model, or invert the image in the latent space of the pretrained model. However, they suffer from two problems: (1) Unsatisfying results for selected regions, and unexpected changes in nonselected regions. (2) They require careful text prompt editing where the prompt should include all visual objects in the input image. To address this, we propose two improvements: (1) Only optimizing the input of the value linear network in the cross-attention layers, is sufficiently powerful to reconstruct a real image. (2) We propose attention regularization to preserve the object-like attention maps after editing, enabling us to obtain accurate style editing without invoking significant structural changes. We further improve the editing technique which is used for the unconditional branch of classifier-free guidance, as well as the conditional one as used by P2P. Extensive experimental prompt-editing results on a variety of images, demonstrate qualitatively and quantitatively that our method has superior editing capabilities than existing and concurrent works.</p>
        </div>
</div>

</div>
</div>


[//]: # (3DI2I)
<div class='paper-box'><div class='paper-box-image'><div><div class="badge">CVPR 2023</div><img src='images/papers/3DI2I.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[3D-Aware Multi-Class Image-to-Image Translation with NeRFs](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_3D-Aware_Multi-Class_Image-to-Image_Translation_With_NeRFs_CVPR_2023_paper.pdf)

**Senmao Li**, Joost van de Weijer, Yaxing Wang, Fahad Shahbaz Khan, Meiqin Liu, Jian Yang

- The first to explore 3D-aware multi-class I2I translation
- Decouple 3D-aware I2I translation into two steps

<div style="display: inline">
        <a href="https://arxiv.org/abs/2303.15012"> [paper]</a>
        <a href="https://github.com/sen-mao/3di2i-translation"> [code]</a>
        <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" >[abstract]</a>
        <div class="abstract"  style="overflow: hidden; display: none;">  
            <p> Recent advances in 3D-aware generative models (3D-aware GANs) combined with Neural Radiance Fields (NeRF) have achieved impressive results for novel view synthesis. However no prior works investigate 3D-aware GANs for 3D consistent multi-class image-to-image (3D-aware I2I) translation. Naively using 2D-I2I translation methods suffers from unrealistic shape/identity change. To perform 3D-aware multi-class I2I translation, we decouple this learning process into a multi-class 3D-aware GAN step and a 3D-aware I2I translation step. In the first step, we propose two novel techniques: a new conditional architecture and a effective training strategy. In the second step, based on the well-trained multi-class 3D-aware GAN architecture that preserves view-consistency, we construct a 3D-aware I2I translation system. To further reduce the view-consistency problems, we propose several new techniques, including a U-net-like adaptor network design, a hierarchical representation constrain and a relative regularization loss. In extensive experiments on two datasets, quantitative and qualitative results demonstrate that we successfully perform 3D-aware I2I translation with multi-view consistency.</p>
        </div>
</div>

</div>
</div>


<ul>

[//]: # (FasterDiffusion)

  <li>
    <img src='https://sen-mao.github.io/FasterDiffusion/' alt="sym" height="100%">
    <a href="https://arxiv.org/abs/2312.09608"> Faster Diffusion: Rethinking the Role of UNet Encoder in Diffusion Models </a>. <strong>Senmao Li</strong>, Taihang Hu, Joost van de Weijier, Fahad Khan, Linxuan Li, Shiqi Yang, Yaxing Wang, Ming-Ming Cheng, Jian Yang. <strong>arXiv</strong>. 
    <div style="display: inline">
        <a href="https://arxiv.org/abs/2312.09608"> [paper]</a>
        <a href="https://sen-mao.github.io/FasterDiffusion/"> [Project Page]</a>
        <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" >[abstract]</a>
        <div class="abstract"  style="overflow: hidden; display: none;">  
            <p> One of the key components within diffusion models is the UNet for noise prediction. While several works have explored basic properties of the UNet decoder, its encoder largely remains unexplored. In this work, we conduct the first comprehensive study of the UNet encoder. We empirically analyze the encoder features and provide insights to important questions regarding their changes at the inference process. In particular, we find that encoder features change gently, whereas the decoder features exhibit substantial variations across different time-steps. This finding inspired us to omit the encoder at certain adjacent time-steps and reuse cyclically the encoder features in the previous time-steps for the decoder. Further based on this observation, we introduce a simple yet effective encoder propagation scheme to accelerate the diffusion sampling for a diverse set of tasks. By benefiting from our propagation scheme, we are able to perform in parallel the decoder at certain adjacent time-steps. Additionally, we introduce a prior noise injection method to improve the texture details in the generated image. Besides the standard text-to-image task, we also validate our approach on other tasks: text-to-video, personalized generation and reference-guided generation. Without utilizing any knowledge distillation technique, our approach accelerates both the Stable Diffusion (SD) and the DeepFloyd-IF models sampling by 41% and 24% respectively, while maintaining high-quality generation performance. </p>
        </div>
    </div>
  </li>

[//]: # (SuppressEOT)

  <li>
    <img src='https://img.shields.io/github/stars/sen-mao/SuppressEOT' alt="sym" height="100%">
    <a href="https://arxiv.org/abs/2402.05375"> Get What You Want, Not What You Don't: Image Content Suppression for Text-to-Image Diffusion Models </a>. <strong>Senmao Li</strong>, Joost van de Weijer, Taihang Hu, Fahad Shahbaz Khan, Qibin Hou, Yaxing Wang, Jian Yang. <strong>ICLR 2024</strong>. 
    <div style="display: inline">
        <a href="https://arxiv.org/abs/2402.05375"> [paper]</a>
        <a href="https://github.com/sen-mao/SuppressEOT"> [code]</a>
        <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" >[abstract]</a>
        <div class="abstract"  style="overflow: hidden; display: none;">  
            <p> The success of recent text-to-image diffusion models is largely due to their capacity to be guided by a complex text prompt, which enables users to precisely describe the desired content. However, these models struggle to effectively suppress the generation of undesired content, which is explicitly requested to be omitted from the generated image in the prompt. In this paper, we analyze how to manipulate the text embeddings and remove unwanted content from them. We introduce two approaches, which we refer to as **soft-weighted regularization** and **inference-time text embedding optimization**. The first regularizes the text embedding matrix and effectively suppresses the undesired content. The second method aims to further suppress the unwanted content generation of the prompt, and encourages the generation of desired content. We evaluate our method quantitatively and qualitatively on extensive experiments, validating its effectiveness. Furthermore, our method is generalizability to both the pixel-space diffusion models (i.e. DeepFloyd-IF) and the latent-space diffusion models (i.e. Stable Diffusion).</p>
        </div>
    </div>
  </li>

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


[//]: # (StyleDiffuison)

  <li>
    <img src='https://img.shields.io/github/stars/sen-mao/StyleDiffusion' alt="sym" height="100%">
    <a href="https://arxiv.org/abs/2312.09608"> StyleDiffusion: Prompt-Embedding Inversion for Text-Based Editing </a>. <strong>Senmao Li</strong>, Joost van de Weijer, Taihang Hu, Fahad Shahbaz Khan, Qibin Hou, Yaxing Wang, Jian Yang. <strong>CVMJ 2024</strong>. 
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
    <a href="https://arxiv.org/abs/2303.15012"> 3D-Aware Multi-Class Image-to-Image Translation with NeRFs </a>. <strong>Senmao Li</strong>, Joost van de Weijer, Yaxing Wang, Fahad Shahbaz Khan, Meiqin Liu, Jian Yang. <strong>CVPR 2023</strong>. 
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
- *Conference Reviewer:* ICLR'25, NeurIPS'24


# 💻 Internships

- *2024.07 - 2024.09*, A research stay, supervised by Dr. [Joost van de Weijer](https://scholar.google.com/citations?user=Gsw2iUEAAAAJ) in [Computer Vision Center](https://www.cvc.uab.es/), Autonomous University of Barcelona, Spain.
