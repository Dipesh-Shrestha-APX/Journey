# 📘 Learning Progress Journal

This document tracks my day-by-day journey in learning Data Science, Machine Learning, and Deep Learning.  
It includes milestones, hands-on practice, and project updates.

---

## 📅 June 25

### 🧮 NumPy — The Foundation of Numerical Computing in Python

- ✅ Completed a **13-video playlist on NumPy**  
  → Practiced core NumPy concepts  
  → _Code available in the `learnings` section_

### 🐼 Pandas — Powerful Data Analysis and Manipulation Tool

- ✅ Completed a **15-video playlist on Pandas**  
  → Practiced data manipulation and analysis  
  → _Code available in the `learnings` section_

---

## 📅 July 1

- 🚀 Trained my **first basic ML model**
- 🌐 Deployed it using **Flask** on a cloud server  
  → Accessible online through a working endpoint

---

## 📅 July 6

### 🔥 PyTorch — Deep Learning Framework that Feels Like Python

- 🧠 Learned the fundamentals of **PyTorch**:
  - Tensors
  - `autograd`
  - `nn.Module`
- 🛠️ Built a basic **training pipeline** using PyTorch

---

## 📅 July 8

### 🦜 LangChain — Build LLM-Powered Applications with Ease

- 🔍 Explored **LangChain components**:
  - Chat Models
  - Embedding Models  
- ✅ Performed **semantic search**:  
  → Compared queries with documents and returned the best match  
  → _Code in the `endToEndProject` section_

- 📦 Learned about `Dataset` and `DataLoader` classes  
  → Updated the PyTorch training pipeline accordingly

---

## 📅 July 9

### 📊 Matplotlib — Create Stunning Visualizations in Python

- 📊 Practiced basic **Matplotlib** plotting and visualization techniques

---

## 📅 July 10

- 🤖 Built a complete **ANN (Artificial Neural Network) training pipeline** from scratch using PyTorch  
- 📈 Dataset: Fashion MNIST  
  → Trained on 6,000 rows → Accuracy: **83.08%**  
  → Trained on full dataset (~48,000 rows) using GPU → Accuracy: **88.67%**

### ⚙️ GPU Optimization Techniques Used:

- Created `device` object using `cuda`
- Moved model and data to GPU
- Increased batch size for efficiency
- Enabled **pin memory** for faster data transfer

---

## 📌 Next Steps

- 🔧 Improve ANN accuracy through hyperparameter tuning
- 🧱 Begin working on CNNs (Convolutional Neural Networks)
- 🔄 Explore transfer learning using pre-trained models

---

## 📅 July 11

- 📂 Learnt about different parameters used in `pd.read_csv()` while working with CSV files.

---

## 📅 July 12

### ⚙️ scikit-learn (sklearn) Essentials

- 📚 Learnt the concept of **Estimator**, which includes both `predictors` and `transformers`, and explored their different types and roles  
  in the scikit-learn framework, including how they interact in building ML workflows.

- ✅ Mastered the foundational concepts required to build and use estimators — including custom transformers, predictors, mixins, pipelines,  
  and column transformations. Also covered the proper usage of `.fit()`, `.transform()`, `.fit_transform()`, and `.predict()` methods within different stages of model development.

---

## 📅 July 13

### 🌐 Prompt Engineering & UI Integration

- 📦 Learnt the basics of how to **pass prompts** to an **API** and a **local server**, understanding the flow from backend to model interaction.  
- 🖥️ Built a simple **Streamlit UI** to send prompts and receive responses, connecting the front-end interface with the backend logic.

---

## 📅 July 14

### 🧪 Optuna & Hyperparameter Optimization with PyTorch 🔥

- 🚀 Learnt what the **Optuna** framework is, and explored how to perform **hyperparameter tuning** across multiple algorithms.  
  Discovered how to:
  - 🔍 Compare different algorithms based on performance metrics like **accuracy**
  - 🧠 Select the **best algorithm** automatically
  - ⚙️ Find the **best parameter values** using Optuna's powerful search and pruning features  

- 🧪 Utilized the learnt hyperparameter tuning to optimize the previously built **ANN model** by exploring different hyperparameters:
  - Layers: number of hidden layers, neurons per layer  
  - Training: `epochs`, `batch_size`, `dropout_rate`  
  - Optimizer settings: `optimizer`, `weight_decay`, `learning_rate`  
  → Ran **3 trials**, and achieved a model accuracy of **88.26%**

### 🧠 ANN Regularization Techniques

- 🔧 Practiced optimizing **Artificial Neural Networks (ANNs)** using:
  - 🧲 **L2 Regularization** to prevent overfitting by penalizing large weights  
  - 🌧️ **Dropout** to randomly deactivate neurons during training and improve generalization

---

### 📅 July 16

🧠 **CNN Learning Step 1: MNIST Classification**

🔹 Built a **LeNet-5 architecture** using 🧪✨ **TensorFlow**  
📊 Achieved **98.74% accuracy** on the **test set**

🔹 Built a **basic CNN pipeline** using 🔥 **PyTorch**  
📈 Achieved **99.93% accuracy** on the **training set**  
📉 Achieved **92.97% accuracy** on the **test set**

---

## 📘 July 17: Transfer Learning & Data Augmentation

🔹 Explored the fundamentals of **Transfer Learning** using 🧪✨ **TensorFlow**  
&nbsp;&nbsp;&nbsp;&nbsp;▫️ **Feature Learning Method**: Used pre-trained CNNs to extract high-level features  
&nbsp;&nbsp;&nbsp;&nbsp;▫️ **Fine-Tuning Method**: Unfroze and retrained upper layers of the model for task-specific tuning

🔹 Learnt and applied **Data Augmentation** 🧪 to expand dataset diversity and improve model generalization

---

## 📘 July 19: Transfer Learning in Practice

🔹 Implemented **Transfer Learning (Feature Learning Method)** using 🔥 **PyTorch**  
📊 Achieved **88.06% test accuracy** on a custom dataset  
✅ Demonstrated the effectiveness of using pre-trained models with limited data

