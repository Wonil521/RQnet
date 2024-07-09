<div align="center">
  
  <h2>RQ-Net for IVIM-DKI quantification (MRM 2023)</h2>
</div>

We provide the training and test code along with the trained weights and the demo dataset used for RQnet. <br>
Our paper was published at Magnetic Resonance in Medicine. <br>
If you find this repository useful, please consider citing our [paper](https://doi.org/10.1002/mrm.29454).

# RQnet for IVIM-DKI
### Registration

<img src="https://github.com/Wonil521/RQnet/assets/59683100/ea72a123-2765-45a5-ad5a-8369788148e4" width="25%" height="25%">*DWIs w/ motion*
<img src="https://github.com/Wonil521/RQnet/assets/59683100/624132f2-5d80-41f4-b61a-0ca97a7604aa" width="25%" height="25%">*Results of RQnet*


### Quantification
<img src="https://github.com/Wonil521/RQnet/assets/59683100/4b49cff6-20d8-455d-bf82-f1bfb908c50c" width="50%" height="50%">

**Reference**:  
> Wonil Lee, Giyong Choi, Jongyeon Lee, Hyunwook Park. Registration and quantification network (RQnet) for IVIM‐DKI analysis in MRI. Magnetic Resonance in Medicine. 2023 Jan;89(1):250-61.
```BibTeX
@article{lee2023registration,
  title={Registration and quantification network (RQnet) for IVIM-DKI analysis in MRI},
  author={Lee, Wonil and Choi, Giyong and Lee, Jongyeon and Park, HyunWook},
  journal={Magnetic Resonance in Medicine},
  volume={89},
  number={1},
  pages={250--261},
  year={2023},
  publisher={Wiley Online Library}
}
```
### Requirements
Our code is implemented using Pytorch.
The code was tested under the following setting:  
* Python 3.12
* CUDA 12.5
* NVIDIA GeForce RTX 4080

## Usage
1. Notebook
- Open RQnet_taining_and_test.ipynb.
- Run each cell.

2. Python
```bash
$ pip install -r requirements.txt
$ python data_gen_digital_phantom.py #for DWI generation
$ python Train_and_Test.py
```

## License
The source codes can be freely used for research and education only. 

## Contact
Please contact me via email (wonil@kaist.ac.kr) for any problems regarding the released code.
