Dated: (26/7/25)

#### General info
Updated and improved pre-processing using preprocess2 dir with aligned faces and all
audios converted to mono channels. 
Dataset: FakeAVCeleb_balanced_V6_500 contains: 
        (397 train sets,50 val sets,49 test sets).

#### Info files in MMD-Net_main dir
1. MMD_datset_v2.py is main custom data loader file.
2. Analyze model.py uses utils/model_analysis.py to generate a sudo simulation of all
    tensors and display their sizes/shapes.
3. Main_v1.py main python file to run MMD-Net_main dir.


Dated: (6/8/25)
#### Info files in MMD-Net_main dir
1. MMD_datset_v4.py is main custom data loader file.(only static features).
2. Analyze model.py uses utils/model_analysis.py to generate a sudo simulation of all
    tensors and display their sizes/shapes.
3. Main_v2.py main python file to run MMD-Net_main dir.
4. Losses_v2.py main loss file.
5. models_v0 previous models file folder (temp + static), now models for static only.


Dated: (22/8/25)
cheked & verified:
    -Efficient Net
    -ResNet
    -MMD_dataset_v4
    -Reconstruction (visual decoder fixed only)
        --'irrelevant': virr_swapped  # to determine/check later
        -- Audio decoder remaining

Dated: (14/9/25)

Dated: (5/10/25) updated on (12/10/25)*
    -Reconstruction (Full fixed)
        --'irrelevant': virr_swapped  # to determine/check later
        -- visual decoder output x= torch.Size([600, 3, 100, 100]) check if should be reshaped? fixed to torch.Size([1, 8, 75, 3, 100, 100]) as 8 videos (original + Reconstructed)
        -- Audio decoder done, now to check shapes and relate to MMD_NET.py flow, what variables are necessary what are not.
    -MMT.py (working)
        -- 
    - shouldnt the video encoder shapes for vfs and vir be B,4,C,75,512 same for audio encoder too? fixed
    - new audio Encoder Freq+TCN based, Visual Encoder improved to with new heads too.
    - now to chng and implment CMAF in MMT.py, also losses remain.