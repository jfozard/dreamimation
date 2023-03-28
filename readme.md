# Dreamimation

Based heavily on the excellent Stable-dreamfusion repository. This code uses a video diffusion model
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



Note that I've swapped the camera coordinate system back to the original NeRF / blender / NeRFStudio
one.

```
@misc{stable-dreamfusion,
    Author = {Jiaxiang Tang},

https://user-images.githubusercontent.com/4390954/228365338-5e3c27e0-ecd3-414a-a513-a9f0500b5b4a.mp4


    Year = {2022},
    Note = {https://github.com/ashawkey/stable-dreamfusion},
    Title = {Stable-dreamfusion: Text-to-3D with Stable-diffusion}
}
```
