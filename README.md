

# Repo for my Individual Project on model-based RL within finance

For libraries and packages:
```bash
pip install -r requirements.txt
```
So far, only experiments on some price prediciton methods have been finished, here are some results:

<p align="center">
<img src="results\Experimet Results\sp100_predicitons.png (1)" height = "250" alt="" align=center />
<br><br>
<b>Figure 1.</b> A sample of S&P100 predicted asset prices using a MLP, Transformer and Autoformer
</p>

<p align="center">
<img src="results\Experimet Results\nasdaq100_predicitons.png (1)" height = "250" alt="" align=center />
<br><br>
<b>Figure 1.</b> A sample of NASDAQ100 predicted asset prices using a MLP, Transformer and Autoformer
</p>

<p align="center">
<img src="results\Experimet Results\csi100_predicitons.png" height = "250" alt="" align=center />
<br><br>
<b>Figure 1.</b> A sample of CSI300 predicted asset prices using a MLP, Transformer and Autoformer
</p>

All datasets are an aggregation of ~80-100 historical close data

<p align="center">
<img src="results\Experimet Results\metric_table.png" height = "250" alt="" align=center />
<br><br>
<b>Figure 1.</b> Table of a performance on a basic strategy and prediction metrics
</p>

The scirpt for these experiments is 
```bash
python predictor_experiments.py
```

# Related Repos:

Autoformer: https://github.com/thuml/Autoformer
FinRL: https://github.com/AI4Finance-Foundation/FinRL

# Related Papers (only from what's been implemented):

Autoformer: https://arxiv.org/abs/2106.13008
FinRL: https://arxiv.org/abs/2011.09607
