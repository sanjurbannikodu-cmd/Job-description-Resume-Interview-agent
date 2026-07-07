# JD-Resume Interview Agent

This agent extracts skills from job descriptions and generates relevant interview questions to help candidates prepare for interviews.

## Features

- Extracts skills from job descriptions using NLP techniques and generates relevant interview questions for interview preparation.

## Features

- Extracts technical skills from job descriptions using spaCy NLP
- Normalizes skills against a predefined taxonomy
- Generates technical, behavioral, and situational interview questions
- Command-line interface for easy use
- Outputs results in JSON format or displays them in the console

## Installation

1. Clone or copy this repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
3. Activate the virtual environment:
   ```bash
   .\venv\Scripts\Activate.ps1  # On Windows
   ```
4. Install required packages:
   ```bash
   pip install spacy transformers sentence-transformers
   ```
5. Download the spaCy English model:
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Usage

### Basic Usage
```bash
python jd_agent_improved.py --file path/to/job_description.txt
```

### With Direct Text Input
```bash
python jd_agent_improved.py --jd "Job description text goes here"
```

### Save Output to JSON File
```bash
python jd_agent_improved.py --file jd.txt --output results.json
```

### Interactive Mode
```bash
python jd_agent_improved.py
```
Then paste your job description and press Ctrl+D (or Ctrl+Z on Windows) when finished.

## Output

The agent provides:
1. **Extracted Skills**: List of technical skills identified from the job description
2. **Generated Questions**:
   - Technical questions: Focus on practical application and problem-solving
   - Behavioral questions: Focus on past experiences and learning approaches
   - Situational questions: Focus on hypothetical scenarios and communication

## Example

Given a job description like:
```
We are looking for a Senior Python Developer with 5+ years of experience in developing web applications using Django and Flask. The ideal candidate should have strong knowledge of RESTful APIs, PostgreSQL database, and AWS cloud services. Experience with Docker containers and Kubernetes orchestration is preferred. Knowledge of React.js and JavaScript for frontend development is a plus. The candidate should be familiar with Git version control and agile/scrum methodologies.
```

The agent will output:
- Extracted skills: agile, django, docker, flask, git, javascript, python, react, sql
- Technical questions about each skill (e.g., "Can you explain your experience with docker and how you've applied it in projects?")
- Behavioral questions for selected skills
- Situational questions for selected skills

## How It Works

1. **Skill Extraction**: Uses spaCy to identify noun chunks, named entities, and skill-related phrases
2. **Skill Normalization**: Maps extracted terms to a predefined skill taxonomy
3. **Question Generation**: Creates questions using templates tailored to each skill type
4. **Output Formatting**: Presents results in a clear, organized manner

## Customization

You can customize the agent by:
- Modifying the skill categories in the `skill_categories` dictionary
- Adding or changing question templates in the `generate_interview_questions` method
- Adjusting the filtering logic in `_is_generic_phrase` and `_normalize_skills` methods

## Running the Web App

Start the Flask server:
```bash
python app.py
```

Running locally (copy/paste):
- http://127.0.0.1:5000/


## Requirements

- Python 3.7+
- spaCy
- transformers
- sentence-transformers
- torch


## License

MIT
