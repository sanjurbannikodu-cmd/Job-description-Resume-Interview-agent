#!/usr/bin/env python3
"""
Flask web application for the JD-Resume Interview Agent
"""

from flask import Flask, render_template, request, jsonify
import json
import os
from jd_agent_improved import JDResumeAgent

app = Flask(__name__)

# Initialize the agent once when the app starts
agent = JDResumeAgent()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        jd_text = data.get('job_description', '').strip()
        
        if not jd_text:
            return jsonify({'error': 'Job description is required'}), 400
        
        # Process the job description
        result = agent.process_jd(jd_text)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze-file', methods=['POST'])
def analyze_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read file content
        jd_text = file.read().decode('utf-8').strip()
        
        if not jd_text:
            return jsonify({'error': 'File is empty'}), 400
        
        # Process the job description
        result = agent.process_jd(jd_text)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
