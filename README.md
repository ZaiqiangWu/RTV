# Real-Time Virtual Try-On

[![arXiv](https://img.shields.io/badge/arXiv-2506.12348-b31b1b.svg)](https://arxiv.org/abs/2506.12348)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/wuzaiqiang/rtv_ckpts)

This project supports real-time virtual try-on without the need for any special sensors. All you need is a webcam and a PC with a GPU equivalent to an RTX 3060 or higher.

Unlike other image-based virtual try-on methods, our approach requires training a dedicated network for each garment in a low-barrier, accessible way (see [this paper](https://arxiv.org/abs/2506.10468)).
Our method simply overlay the synthesized garment on the top of human body without removing the original garment to achieve real-time performace.

![Demo GIF](assets/output.gif)

Check our [tight garment demo](https://www.youtube.com/watch?v=dub0pJ2kXA4) and [loose garment demo](https://www.youtube.com/watch?v=7hm1yBsFzHc) on YouTube.


## Installation

```
git clone https://github.com/ZaiqiangWu/RTV.git
cd RTV
```

### Environment
```
sudo apt install gcc g++ libxcb-xinerama0-dev libxcb-icccm4-dev libxcb-image0-dev libxcb-keysyms1-dev libxcb-randr0-dev libxcb-shape0-dev libxcb-sync-dev libxcb-xfixes0-dev libxcb-xkb-dev 
sudo apt install qtbase5-dev qtbase5-dev-tools libqt5gui5 libqt5widgets5 libqt5multimedia5 libqt5multimediawidgets5 libqt5multimedia5-plugins libpulse-mainloop-glib0
conda create -n rtv python=3.9
conda activate rtv
pip install -r requirements.txt
pip install detectron2@git+https://github.com/facebookresearch/detectron2.git
pip install git+https://github.com/ZaiqiangWu/ROMP.git#subdirectory=simple_romp
```

### Weights
Download our pretrained checkpoints by running the following command. Please refer to [this page](./training_instructions.md) for instructions on training your own garment items.
```
sudo apt install git-lfs
git lfs install
git clone https://huggingface.co/wuzaiqiang/rtv_ckpts
```

### Real-time virtual try-on
Please make sure that there is a webcam connected to your PC before running the following command.
```
python rtl_demo.py
```

## BibTeX
```text
@inproceedings{wu2025real,
  title={Real-Time Per-Garment Virtual Try-On with Temporal Consistency for Loose-Fitting Garments},
  author={Wu, Zaiqiang and Shen, I-Chao and Igarashi, Takeo},
  booktitle={Computer Graphics Forum},
  pages={e70272},
  year={2025},
  organization={Wiley Online Library}
}

@misc{wu2025lowbarrier,
    title={Low-Barrier Dataset Collection with Real Human Body for Interactive Per-Garment Virtual Try-On},
    author={Zaiqiang Wu and Yechen Li and Jingyuan Liu and Yuki Shibata and Takayuki Hori and I-Chao Shen and Takeo Igarashi},
    year={2025},
    eprint={2506.10468},
    archivePrefix={arXiv},
    primaryClass={cs.GR}
}
```

## License
This project is licensed under a custom Apache 2.0-based license with a **Non-Commercial Use** clause.  
You may use, modify, and share the code freely for non-commercial purposes.  
Commercial use requires prior written permission from the author.  
See the [LICENSE](./LICENSE) file for details.
