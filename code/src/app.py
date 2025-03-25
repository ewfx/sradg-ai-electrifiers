from flask import Flask, request, send_file, jsonify
import pandas as pd
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS
import logging
from sklearn.ensemble import IsolationForest
import numpy as np

app = Flask(__name__)
CORS(app)  

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER
ALLOWED_EXTENSIONS = {"xlsx"}

def allowed_file(filename):
    """Check if the uploaded file is an Excel file."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_anomalies(file_path):
    logging.info(f"Processing file: {file_path}")

    try:
        df = pd.read_excel(file_path)
        df.columns = df.columns.str.strip()  
        df['As of Date'] = pd.to_datetime(df['As of Date'])  
        df = df.sort_values(by=["Account", "As of Date"])  
       
        def detect_for_account(account_df):
            account_df["Previous Balance"] = account_df["Balance Difference"].shift(1)
            account_df["Balance Change"] = abs(account_df["Balance Difference"] - account_df["Previous Balance"])
            
            account_df["Balance Change"].fillna(0, inplace=True)
           
            model = IsolationForest(contamination=0.05, random_state=42)
            account_df["Anomaly Score"] = model.fit_predict(account_df[["Balance Change"]])
           
            def identify_anomaly(row):
                if row["Anomaly Score"] == -1:
                    return "Anomaly"
                if abs(row["Balance Difference"] - row["Previous Balance"]) > (account_df["Balance Change"].median()):
                    return "Anomaly"
                return "Normal"

            account_df["Anomaly Detection"] = account_df.apply(identify_anomaly, axis=1)
          
            def add_comments(row):
                if row["Anomaly Detection"] == "Anomaly":
                    if abs(row["Balance Difference"] - row["Previous Balance"]) > (5 * account_df["Balance Change"].median()):
                        return "Sudden Spike/Drop in Balance Detected"
                    return "Unusual Transaction Pattern"
                return ""

            account_df["Comments"] = account_df.apply(add_comments, axis=1)

            return account_df

        df = df.groupby("Account", group_keys=False).apply(detect_for_account)

        logging.info("ML-Based Anomaly detection completed successfully.")
        return df.drop(columns=["Previous Balance", "Balance Change", "Anomaly Score"])

    except Exception as e:
        logging.error(f"Error in anomaly detection: {str(e)}")
        raise

@app.route("/upload", methods=["POST"])
def upload_file():
    logging.info("Received a file upload request.")

    if "file" not in request.files:
        logging.warning("No file found in request.")
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    
    if file.filename == "":
        logging.warning("No file selected.")
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)
        logging.info(f"File {filename} saved successfully.")

        try:
            output_df = detect_anomalies(file_path)
            
            output_filename = f"Processed_{filename}"
            output_path = os.path.join(app.config["OUTPUT_FOLDER"], output_filename)
            output_df.to_excel(output_path, index=False)
            logging.info(f"Processed file saved: {output_path}")
            
            return jsonify({"download_url": f"http://127.0.0.1:5000/download/{output_filename}"})

        except Exception as e:
            logging.error(f"Processing failed: {str(e)}")
            return jsonify({"error": f"Processing failed: {str(e)}"}), 500

    logging.warning("Invalid file format received.")
    return jsonify({"error": "Invalid file format"}), 400

@app.route("/download/<filename>", methods=["GET"])
def download_file(filename):
    file_path = os.path.join(app.config["OUTPUT_FOLDER"], filename)

    logging.info(f"Download requested for: {filename}")
    logging.info(f"Expected file path: {file_path}")

    if os.path.exists(file_path):
        logging.info(f"File {filename} found. Sending for download.")
        return send_file(file_path, as_attachment=True)
    else:
        logging.error(f"File {filename} not found.")
        return jsonify({"error": "File not found"}), 404

if __name__ == "__main__":
    logging.info("Starting Flask server...")
    app.run(debug=True, use_reloader=False, host="0.0.0.0")
