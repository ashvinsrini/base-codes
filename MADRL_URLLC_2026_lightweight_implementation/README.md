\# Asynchronous Multi-Agent DRL for Interference-Aware Resource Scheduling



This repository contains the core Python implementation for an asynchronous multi-agent deep reinforcement learning pipeline for radio resource scheduling in mobile in-X subnetworks. The workflow combines:



\- a \*\*pretrained LSTM\*\* for one time step ahead prediction,

\- an \*\*asynchronous multi-agent DDPG scheduler\*\* for decentralized channel allocation under URLLC-style reliability constraints, and

\- \*\*confidence-interval based evaluation scripts\*\* for the main plots used in the study.



\---



\## Repository structure



```text

.

├── DRL\_async\_ci\_train.py

├── LSTM\_predict.py

├── SINR\_cdf\_CI\_runner.py

├── slurm.sh

├── slurm\_DRL\_async.sh

├── slurm\_LSTM.sh

├── DRL\_aysnch\_ci.ipynb

├── LSTM\_interfere.ipynb

├── SINR\_cdf\_refactore.ipynb

├── Utils/

│   ├── env\_orig.py

│   └── LSTM.py

└── sample figures / output folders

```



\---



\## What each file does



\### Main Python scripts



\#### `LSTM\_predict.py`

This script trains the LSTM model that is later used by the asynchronous DRL pipeline.



It:



\- simulates the wireless environment,

\- generates interference(assuming frequency reuse 1 during pretraining) from channel traces,

\- trains the LSTM over multiple runs,

\- computes 95% confidence intervals,

\- saves the training-loss and target-vs-predicted trace figures,

\- stores processed arrays for later plotting, and

\- saves the trained checkpoint `LSTM\_state\_dict.pth`.



Important outputs created by this script include:



\- `training\_mse\_loss\_ci.png`

\- `training\_mse\_loss\_ci.pdf`

\- `interference\_trace\_ci.png`

\- `interference\_trace\_ci.pdf`

\- `train\_loss\_runs.npy`

\- `pred\_test\_runs.npy`

\- `tgt\_test\_runs.npy`

\- `processed\_plot\_data.npz`

\- `metadata.json`

\- `LSTM\_state\_dict.pth`



By default, the script writes these outputs to:



```text

lstm\_interfere\_ci\_fixed\_outputs/

```



\---



\#### `DRL\_async\_ci\_train.py`

This is the core asynchronous multi-agent DRL training script.



It:



\- loads the pretrained LSTM,

\- builds the wireless environment,

\- trains the asynchronous multi-agent scheduler over several runs,

\- computes the BLER CDF with 95% confidence intervals, and

\- saves both the figure and the underlying arrays.



Typical outputs created by this script include:



\- `async\_bler\_cdf\_ci.png`

\- `bler\_runs\_raw.npz`

\- `bler\_cdf\_ci\_data.npz`



By default, the script writes these outputs to:



```text

async\_8agent\_bler\_ci\_outputs/

```



\*\*Important:\*\* this script expects a pretrained LSTM checkpoint and imports the utility files from `Utils/`. In the current version, the checkpoint path is set through `LSTM\_MODEL\_DIR` and `LSTM\_MODEL\_NAME`, so one should either:



1\. update the checkpoint path inside `DRL\_async\_ci\_train.py`, or  

2\. copy the trained `LSTM\_state\_dict.pth` to the location expected by the script.



\---



\#### `SINR\_cdf\_CI\_runner.py`

This script generates the Monte Carlo SINR CDF results with confidence intervals.



It:



\- recreates the three CDF-style curves used to characterize the channel statistics,

\- computes confidence intervals across runs, and

\- saves the figures, raw arrays, and processed plotting data.



Typical outputs created by this script include:



\- `sinr\_cdf\_ci.png`

\- `sinr\_cdf\_ci.pdf`

\- `sinr\_cdf\_ci\_processed.npz`

\- `sinr\_db\_samples.npy`

\- `mean\_samples.npy`

\- `diff\_samples.npy`

\- `sinr\_cdf\_ci\_metadata.json`



By default, the script writes these outputs to:



```text

sinr\_cdf\_ci\_outputs\_fixed/

```



\---



\### Utility files



\#### `Utils/env\_orig.py`

This file contains the wireless environment implementation used by the asynchronous DRL pipeline. It includes the mobility model, fading and path-loss generation, SINR / reward computation, BLER approximation, and helper functions for environment interaction.



\#### `Utils/LSTM.py`

This file contains the `LSTMModel` class used by the training and inference scripts. It defines the PyTorch LSTM architecture used for one-step prediction of interference / SINR related traces.



\---



\### Notebook files



\#### `LSTM\_interfere.ipynb`

Notebook version of the LSTM training and analysis workflow.



\#### `DRL\_aysnch\_ci.ipynb`

Notebook version of the asynchronous DRL training and confidence-interval analysis workflow.



\#### `SINR\_cdf\_refactore.ipynb`

Notebook version of the SINR CDF analysis workflow.



These notebooks are useful for interactive experimentation, while the `.py` scripts are the streamlined versions for batch execution.



\---



\### Slurm job scripts



\#### `slurm\_LSTM.sh`

Runs the LSTM training script on a cluster environment using:



```bash

python LSTM\_predict.py

```



\#### `slurm\_DRL\_async.sh`

Runs the asynchronous DRL training script using:



```bash

python DRL\_async\_ci\_train.py

```



\#### `slurm.sh`

Runs the SINR CDF analysis using:



```bash

python SINR\_cdf\_CI\_runner.py --num-runs 5 --output-dir SINR\_cdf\_CI\_runner

```



\---



\## Requirements



The codebase is written in \*\*Python\*\* and uses \*\*PyTorch\*\* for the learning-based components. FOr the scripts, the main packages used are:



```bash

pip install numpy scipy pandas matplotlib tqdm pillow plotly

```



one will also need a working \*\*PyTorch\*\* installation that matches the machine or cluster setup.



A simple example is:



```bash

pip install torch

```



\---



\## Suggested folder layout



Place the utility files inside a `Utils/` directory:



```text

project\_root/

├── DRL\_async\_ci\_train.py

├── LSTM\_predict.py

├── SINR\_cdf\_CI\_runner.py

├── Utils/

│   ├── env\_orig.py

│   └── LSTM.py

```



This is required because `DRL\_async\_ci\_train.py` imports:



```python

from Utils.env\_orig import env

from Utils.LSTM import LSTMModel

```



\---



\## Recommended run order



\### 1. Train the LSTM predictor



```bash

python LSTM\_predict.py

```



This produces the trained checkpoint and the LSTM-related confidence interval plots.



\### 2. Update the LSTM checkpoint path if needed



Open `DRL\_async\_ci\_train.py` and make sure `LSTM\_MODEL\_DIR` and `LSTM\_MODEL\_NAME` point to the checkpoint produced in Step 1.



\### 3. Run the asynchronous DRL pipeline



```bash

python DRL\_async\_ci\_train.py

```



This produces the BLER CDF figure and the saved confidence-interval arrays.



\### 4. Run the SINR CDF analysis



```bash

python SINR\_cdf\_CI\_runner.py --num-runs 5 --output-dir SINR\_cdf\_CI\_runner

```



This produces the SINR CDF figure and associated saved arrays.



\---



\## Example outputs



sample PNG files which are included in the repository are the type of outputs oneould expect.



\### Training MSE loss with 95% CI



```markdown

!\[Training MSE loss with 95% CI](training\_mse\_loss\_ci.png)

```



\### Target vs predicted interference power with 95% CI



```markdown

!\[Target vs predicted interference power with 95% CI](interference\_trace\_ci.png)

```



\### Asynchronous BLER CDF with 95% CI



```markdown

!\[Asynchronous BLER CDF with 95% CI](async\_bler\_cdf\_ci.png)

```



\### SINR CDF with confidence intervals



```markdown

!\[SINR CDF with confidence intervals](sinr\_cdf\_ci.png)

```



\---



\## Notes



\- The asynchronous DRL script depends on a pretrained LSTM checkpoint, so run `LSTM\_predict.py` first.

\- The main environment and LSTM utility files should remain inside `Utils/` for the imports to work as written.

\- The scripts save both figures and raw arrays, which makes it easy to regenerate plots later without retraining.

\- Cluster execution examples are already provided through the included Slurm scripts.



\---



\## Related papers



If you use this code in academic work, please cite the associated papers:



1\. \*\*Asynchronous Multi-Agent Reinforcement Learning for Scheduling in Subnetworks\*\*

2\. \*\*Multi-Agent Reinforcement Learning Approach Scheduling for In-X Subnetworks\*\*



\---




\## Recommended extra files for the repository



\## Acknowledgement



This code accompanies research on multi-agent learning for URLLC-oriented scheduling in mobile in-X subnetworks.

