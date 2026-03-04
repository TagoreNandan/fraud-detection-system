## System Architecture

The fraud detection platform is built using a microservice architecture where a FastAPI backend serves model predictions and a Streamlit dashboard provides monitoring and visualization.


graph TD

A[Transaction Source<br/>- Transaction Simulator<br/>- CSV Upload<br/>- Manual Input]

B[FastAPI Backend<br/>/predict endpoint]

C[Fraud Detection Model<br/>XGBoost Pipeline]

D[Prediction + SHAP Explanation]

E[SQLite Database<br/>Transaction Logs]

F[Streamlit Dashboard<br/>Monitoring + Visualization]

A --> B
F --> B
B --> C
C --> D
D --> E
E --> F
