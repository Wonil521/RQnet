<div align="center">
  
  <h2>RQ-Net for IVIM-DKI quantification (MRM 2023)</h2>
</div>

We provide the training and test code along with the trained weights and the demo dataset used for RQnet.
Our paper was published at Magnetic Resonance in Medicine.
If you find this repository useful, please consider citing our [paper](https://doi.org/10.1002/mrm.29454).

# RQnet for IVIM-DKI
### Registration


### Quantification



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
