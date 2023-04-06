# Dreamimation

Based heavily on the excellent Stable-dreamfusion repository. 
https://github.com/ashawkey/stable-dreamfusion

This code uses a video diffusion model
https://huggingface.co/spaces/damo-vilab/modelscope-text-to-video-synthesis
to optimize the time-dependent deformation of a NeRF. 
This approach follows Make-A-Video 3D, but deforming the tripane NeRF rather than having it evolve in time.

Still very much a work-in progress.

Sample starting timepoint data at

https://drive.google.com/file/d/16NtLqgJKUA9y5N2m-Yvx057V5r0F-Zgu/view?usp=share_link

To use:
```
python main_vid.py --workspace corgi --static corgi.pth
```

Sample results

https://user-images.githubusercontent.com/4390954/228365338-5e3c27e0-ecd3-414a-a513-a9f0500b5b4a.mp4
(^This was from previous version of code)

For horse demo
```
python main_vid.py --d_res 32 --fp16 --workspace horse_stretch_64 --blob_radius 0.2 --d_scale 0.1 --text "A horse galloping, slow motion, smooth, high quality" --lambda_stretch 0.01 --lambda_emptiness 1.0 --lambda_tv 1e-12 --static df_horse_clip.pth
```
for 64 epochs. Then increase the resolution of the deformation field and run for 440 epochs
```
python main_vid.py --d_res 64 --fp16 --workspace horse_stretch_64_2 --blob_radius 0.2 --d_scale 0.1 --text A horse galloping, slow motion, smooth, high quality --lambda_stretch 0.01 --lambda_emptiness 1.0 --lambda_tv 1e-12 --ckpt horse_stretch_64/checkpoints/df_ep0064.pth
```

Checkpoints can be found at https://drive.google.com/drive/folders/1DRbKCGNBncbVhtUUljKkw7EGUQKG3V5S?usp=share_link

Rendered results

https://user-images.githubusercontent.com/4390954/229761773-78339280-bcd0-41ab-96e1-4bda89b05498.mp4


Note that I've swapped the camera coordinate system back to the original NeRF / blender / NeRFStudio
one.


