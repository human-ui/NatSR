# Natural and Realistic Single Image Super-Resolution

A model that upsamples a given image by 4x.

This repository contains the inference part of the NatSR model described here:

Jae Woong Soh, Gu Yong Park, Junho Jo, and Nam Ik Cho. **[Natural and Realistic Single Image Super-Resolution with Explicit Natural Manifold Discrimination.](http://openaccess.thecvf.com/content_CVPR_2019/html/Soh_Natural_and_Realistic_Single_Image_Super-Resolution_With_Explicit_Natural_Manifold_CVPR_2019_paper.html)** *CVPR 2019*

## How to run

```
from nat_sr import Upsampler

upsampler = Upsampler()
upsampler('my_image.png', 'my_image_4x.png')
```


