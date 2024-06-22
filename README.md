# TCTL-for-cross-load-fault-diagnosis
This is Python code for our paper. The CWRU dataset is too large, please download it yourself if you are interested！

For JNU dataset (data_jnu): The ReadData_JNU_5000.py is for sampling. The CNN_J.py is the model for jnu dataset. The mmd.py is used for domain alignment（MK-MMD） and the TCTL_jnu.py is for test model.

For CWRU dataset (data_cwru): The ReadData200.py is for sampling. The CNN_C.py is the model for cwru dataset. The mmd.py is used for domain alignment（MK-MMD） and the TCTL_cwru.py is for test model.

<!-- 代码使用 -->
If you want to use this code, please
- download datasets (cwru and jnu) in the code dir, and rename their folder name to `data_cwru` and `data_jnu`.
- if you want to run model in cwru, run `python TCTL_cwru.py --src 0 --tar 1`, where `--src` can be chosen in 0, 1, 2, 3, and `--tar` can be chosen in 0, 1, 2, 3. Notice that `--src` != `--tar`!
- if you want to run model in jnu, run `python TCTL_jnu.py --src 600 --tar 800`, where `--src` can be chosen in 600, 800, 1000, and `--tar` can be chosen in 600, 800, 1000. Notice that `--src` != `--tar`!

I hope my code proves helpful to you. Please feel free to reach out if you have any questions!

The citation format: 

Zheng J, Jiang B, Yang C. Proportional periodic sampling for cross-load bearing fault diagnosis[J]. International Journal of Machine Learning and Cybernetics, 2024: 1-13.

@article{zheng2024proportional,
  title={Proportional periodic sampling for cross-load bearing fault diagnosis},
  author={Zheng, Jianbo and Jiang, Bin and Yang, Chao},
  journal={International Journal of Machine Learning and Cybernetics},
  pages={1--13},
  year={2024},
  publisher={Springer}
}

Postscript：The reference work associated with this work is our previous MDPS at https://github.com/IWantBe/MDPS
