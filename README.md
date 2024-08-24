# CCFG-Net: Extremely Fine-Grained Visual Classification over Resembling Glyphs in the Wild.

In summary, the main contributions of this paper are as follows:

- We build benchmark datasets for extremely fine-grained recognition over resembling glyphs in the wild, which are RCC-FGV for resembling Chinese characters discrimination, and EL-FGVC for resembling English letters identification. 

- We demonstrate that recognizing Chinese characters in the wild is an intermediate task between classical categorization and fine-grained classification.

- We devise a deep architecture for this task, incorporating classification and contrastive learning in both Euclidean and angular spaces, in which contrastive learning is applied at the supervised (one against many) and pairwise (one versus one) levels.

- We evaluate our approach on the RCC-FGVC and EL-FGVC datasets using five different encoders/backbones, and  provide comparisons with representative fine-grained classification methods, which demonstrate the effectiveness of our proposed approach in the extremely fine-grained recognition task of discriminating resembling glyphs in the wild.



## Implementation:
#### PDAtt-Unet architecture and Hybrid loss function:
``` Architectures.py ``` contains our implementation of the comparison CNN baseline architectures  (Unet, Att-Unet and Unet++) and the proposed PDAtt-Unet. architecture.

``` Hybrid_loss.py ``` contains the proposed Edge loss function.

#### Training and Testing Implementation:
``` detailed train and test ``` contains the training and testing implementation.

- First: the dataset should be prepared using ``` prepare_dataset.py ```, this saves the input slices, lung mask, and infection mask as ``` .pt ``` files
The datasets could be donwloaded from: http://medicalsegmentation.com/covid19/

- Second:  ``` train_test_PDAttUnet.py ``` can be used to train and test the proposed PDAtt-Unet architecture with the proposed Hybrid loss function (with Edge loss).


## Citation: If you found this Repository useful, please cite:

```bash
Coming Soon
```
