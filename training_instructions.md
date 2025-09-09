# Instructions on Training a New Garment Checkpoint

## Items Required
- A camera capable of recording video in 4K resolution (most smartphones are sufficient).
- The garment intended for virtual try-on.
- A person to wear the garment during recording.

## Step 1
Recording a video of a person wearing the garment following the instruction of [this paper](https://arxiv.org/abs/2506.10468).
The predefined poses can be find [here](assets/pose_guidance/symmetric.pdf).
Please note that you don't need to strictly follow the predefined poses (the more diverse the poses, the better).
We provide an example of a recorded video [here](https://huggingface.co/datasets/wuzaiqiang/Per-GarmentDataset/blob/main/example_video.mp4).

## Step 2
Performing garment segmentation to the recorded video.
there are many available methods. We recommend [this repository](https://github.com/heyoeyo/muggled_sam), which requires only minimal interaction to achieve desirable garment segmentation results.

<img src="assets/demo/Screenshot.png" alt="Description" height="500" />


Exporting the segmentation results as a tar file. We provide an example of a tar file [here](https://huggingface.co/datasets/wuzaiqiang/Per-GarmentDataset/blob/main/example_video/000_obj1_0_to_2135_frames.tar).

## Step 3
Create a directory and extract the tar file inside this directory.

Work in progress ...