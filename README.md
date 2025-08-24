# 🎬 Movie Success Prediction and Sentiment Study  

## 📌 About the Project  
This project explores how **movie critic reviews** and **movie metadata** can be combined to predict a film’s success.  
It leverages **sentiment analysis** and **machine learning models** to analyze review sentiments, study their impact on movie performance, and build predictive models for success classification.  

By doing this, the project not only provides insights into how **audience/critic sentiment affects movies** but also helps in **predicting success using key features like budget, genres, popularity, and sentiments**.  

---

## 🎯 Objective  
- To analyze critic reviews and extract meaningful sentiment scores.  
- To combine sentiments with movie metadata for deeper insights.  
- To build predictive models for determining **movie success or failure**.  
- To visualize trends in sentiments across genres and other factors.  

---

## ✨ Features  
- 🔹 Sentiment analysis of Rotten Tomatoes critic reviews  
- 🔹 Genre-wise sentiment distribution and averages  
- 🔹 Data cleaning, normalization, and feature engineering  
- 🔹 Predictive modeling using Logistic Regression & Random Forest  
- 🔹 Confusion Matrix, ROC Curve, and Classification Report for model evaluation  
- 🔹 Export of results & visualizations (CSV and PNG)  

---

## 📂 Files Included  
- **Movie_Success_Sentiment_Study.ipynb** → Main Google Colab notebook  
- **datasets/** (Can be downloaded from the links given)  
  - RottenTomatoesCriticReviews.csv  
  - MoviesMetadata.csv  
- **outputs/** (CSV results like feature importance, genre sentiment, metrics)  
- **Sentiment Visuals/** (visualization PNGs: ROC curve, confusion matrix, sentiment plots, feature importance, etc.)  
- **Predictive_Model_Summary.pdf** (summary of predictive models used and results)  

---

## 🛠️ Tools & Requirements  
- **Google Colab / Jupyter Notebook**  
- **Python 3.x**  
- Libraries:  
  - `pandas`, `numpy`  
  - `matplotlib`, `seaborn`  
  - `scikit-learn`  
  - `nltk` / `textblob`  

---

## 📂 Datasets Used  
1. **Rotten Tomatoes Critic Reviews**
   [RottenTomatoesCriticReviews.csv](https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset)
2. **Movies Metadata**
   [MoviesMetadata.csv](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)  
---

## 🗂️ Workflow (Steps)  
1. 🔧 **Set up environment** in Google Colab  
2. 📥 **Load datasets** from provided links  
3. 🧹 **Clean and preprocess** data (missing values, normalization, merge datasets)  
4. 💬 **Perform sentiment analysis** on critic reviews  
5. 📊 **EDA & Visualizations**:  
   - Sentiment distribution plots  
   - Genre-wise average sentiment  
   - Strip plots for predicted vs actual success  
6. ⚡ **Feature engineering**: join sentiment with metadata, normalize values  
7. 🤖 **Model building**: Logistic Regression, Random Forest  
8. ✅ **Model evaluation**: Confusion matrix, ROC, classification report  
9. 📤 **Export results** to CSV & save visuals  

---

## 🤖 Models Used  
- Logistic Regression  
- Random Forest Classifier  

---

## 📊 Predictive Model Summary  
A separate PDF (**Predictive_Model_Summary.pdf**) is included in this repo that:  
- Explains the predictive models used  
- Shows evaluation metrics (accuracy, precision, recall, F1-score)  
- Highlights key findings from confusion matrices, ROC, and feature importance  

---

## 🌟 Results & Highlights  
- Positive/negative review patterns clearly identified across genres  
- Genre sentiment analysis showed trends in critic reviews by category  
- Random Forest outperformed Logistic Regression in predictive accuracy  
- Feature importance highlighted **budget, popularity, and sentiment** as key predictors  

---

## 🚀 Future Improvements  
- Use **deep learning models (LSTM/Transformers)** for advanced sentiment analysis  
- Include **audience reviews** along with critic reviews  
- Hyperparameter tuning for better predictive accuracy  
- Deploy as a **web app** for user interaction  

---

## 👤 Author  
**Pasham Tejaswini**  
Movie Success Prediction and Sentiment Study Project  
