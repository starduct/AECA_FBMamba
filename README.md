# AECA-FBMamba: A Framework with Adaptive Environment Channel Alignment and Mamba Bridging Semantics and Details

## Training Instructions

* **Our code framework is based on [Paraformer](https://github.com/LiZhuoHong/Paraformer), and the setup of the training environment can be guided by its README file. After that, to train and test the AECA-FBMamba on the default Chesapeake Bay dataset, follow these steps:**

1. Download our code. We have modified the code "train.py" and "test.py" in the Paraformer. We have also submitted the "multi_eval.py" code to evaluate the model's performance.

2. Run the "Train" command:

   ```bash
   python train.py --dataset Chesapeake --batch_size 4 --max_epochs 100 --savepath *save path of your folder* --gpu 0
   ```

2. After training, run the "Test" command:

   ```bash
   python test.py --dataset Chesapeake --model_path *The path of trained .pth file* --save_path *To save the inferred results* --gpu 0
   ```

3. To evaluate the inferred results, run:

   ```bash
   python python multi_eval.py --result_dir *Path to the inferred results*
   ```

* **To train and test the framework on your dataset:**

1. Generate a train and test lists (.csv) of your dataset (an example is in the "dataset" folder).
2. Change the label class and colormap in the "utils.py" file.
3. Add your dataset_config in the "train.py", "test.py"and "multi_eval.py" files.
4. Run the command above by changing the dataset name.

The Chesapeake Dataset
-------

<img src="https://github.com/starduct/AECA_FBMamba/blob/main/img/The%20Chesapeake%20Dataset.png" width="70%">

The Chesapeake Dataset sample images used above is from the publicly available [**Paraformer GitHub repository**](https://github.com/LiZhuoHong/Paraformer).

The Poland Dataset
-------

The Poland Dataset sample images used above is from the publicly available [**Paraformer GitHub repository**](https://github.com/LiZhuoHong/Paraformer).

<img src="https://github.com/starduct/AECA_FBMamba/blob/main/img/The%20Poland%20dataset.png" width="70%">

The sample images used in the paper include:
1. m_4207417_ne_18_1:
<img src="https://github.com/starduct/AECA_FBMamba/blob/main/img/m_4207417_ne_18_1.png" width="70%">
2. m_4207421_sw_18_1:
<img src="https://github.com/starduct/AECA_FBMamba/blob/main/img/m_4207421_sw_18_1.png" width="70%">
3. m_4207450_nw_18_1:
<img src="https://github.com/starduct/AECA_FBMamba/blob/main/img/m_4207450_nw_18_1.png" width="70%">
4. m_4207706_se_18_1:
<img src="https://github.com/starduct/AECA_FBMamba/blob/main/img/m_4207706_se_18_1.png" width="70%">
5. m_4207716_sw_18_1:
<img src="https://github.com/starduct/AECA_FBMamba/blob/main/img/m_4207716_sw_18_1.png" width="70%">
6. m_4307417_sw_18_1:
<img src="https://github.com/starduct/AECA_FBMamba/blob/main/img/m_4307417_sw_18_1.png" width="70%">
