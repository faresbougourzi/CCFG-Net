# CCFG-Net: Extremely Fine-Grained Visual Classification over Resembling Glyphs in the Wild.

In summary, the main contributions of this paper are as follows:

- We build benchmark datasets for extremely fine-grained recognition over resembling glyphs in the wild, which are RCC-FGV for resembling Chinese characters discrimination, and EL-FGVC for resembling English letters identification. 

- We demonstrate that recognizing Chinese characters in the wild is an intermediate task between classical categorization and fine-grained classification.

- We devise a deep architecture for this task, incorporating classification and contrastive learning in both Euclidean and angular spaces, in which contrastive learning is applied at the supervised (one against many) and pairwise (one versus one) levels.

- We evaluate our approach on the RCC-FGVC and EL-FGVC datasets using five different encoders/backbones, and  provide comparisons with representative fine-grained classification methods, which demonstrate the effectiveness of our proposed approach in the extremely fine-grained recognition task of discriminating resembling glyphs in the wild.

![DesNet_2030](https://github.com/user-attachments/assets/7b4d781a-6638-4ad8-a7ec-1cf323d173b5)


<p align="center">
  Figure 1: Our proposed CCFG-Net approach.
</p> 

## Implementation:
#### PDAtt-Unet architecture and Hybrid loss function:
``` create_train_val_test.py ``` prepare the three splits before training.

``` create_database.py ``` prepares the training data to train our approach.

#### Training and Testing Implementation:
``` train_test_Resnet_SCL.py ``` contains the training of the pretrained backbone with SCL.
``` train_test_CCFG_Net.py ``` contains the training and evaluation of our approach.




## Citation: If you found this Repository useful, please cite:

```bash
Coming Soon
```
