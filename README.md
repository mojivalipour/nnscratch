A Simple Numpy Neural Network Framework


TA: Mojtaba Valipour

CS 486-686 Course Assignment : University of Waterloo - Alice Gao - Spring 2020

- Colab Version: https://colab.research.google.com/github/mojivalipour/nnscratch/blob/master/CS486_686_A2Q2ANN.ipynb
- Local Jupyter Notebook: https://github.com/mojivalipour/nnscratch/blob/master/CS486-686_A2Q2ANN_LOCAL.ipynb
- Python File: https://github.com/mojivalipour/nnscratch/blob/master/CS486-686_A2Q2ANN.py

Dataset:
https://archive.ics.uci.edu/ml/datasets/Anuran+Calls+%28MFCCs%29


Need to fix the bug in relu prediction, a possible solution is: 
```python
elif self.config[self.configKeyList[-1]][-1] == 'relu': 
          yPred = np.round(yPred)
          yPred = np.clip(yPred, 0, self.numClass-1) # sanity check 
```
