# 🚀 Running This Project in Google Colab

This repository contains a battlefield simulation project. Follow the steps below to **clone, install dependencies, and run** the project in **Google Colab**.

---

## **📌 Step 1: Open Google Colab**
1. Go to [Google Colab](https://colab.research.google.com/).
2. Click **“File” → “New Notebook”** to create a new notebook.

---

## **📌 Step 2: Clone the Repository**
Run the following command in a Colab code cell to download the project:
```python
!git clone https://github.com/akcone2003/psych186b_final.git
```
Then, navigate into the project folder:
```python
%cd psych186b_final
```

## 📌 Step 3: Install Required Dependencies
This project includes a requirements.txt file. Install all required packages by running:

```python
!pip install -r requirements.txt
```

## 📌 Step 4: Run the Main Script
To execute the project, run:

```python
!python main.py
```

### **🔹 Pushing Changes Back to GitHub**
If you make changes and want to save them to GitHub, configure Git with **your own GitHub credentials**:
```python
!git config --global user.email "your-email@example.com"  # Replace with your own email
!git config --global user.name "YourGitHubUsername"  # Replace with your own GitHub username
```

Then, commit and push your changes:
```python
!git add .
!git commit -m "Updated results"
!git push origin main
```


