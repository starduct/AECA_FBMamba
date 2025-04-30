# AECA-FBMamba: A Framework with Adaptive Environment Channel Alignment and Mamba Bridging Semantics and Details

## Training Instructions

* **Our code framework is based on Paraformer, and the setup of the training environment can be guided by its README file. After that, to train and test the AECA-FBMamba on the default Chesapeake Bay dataset, follow these steps:**

1. Run the "Train" command:

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
