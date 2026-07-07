#!/usr/bin/env python3
"""
JD-Resume Interview Agent
Extracts skills from Job Description and generates interview questions
"""

import spacy
import re
import json
import argparse
from typing import List, Dict, Set
from sentence_transformers import SentenceTransformer, util
import torch

class JDResumeAgent:
    def __init__(self):
        """Initialize the agent with spaCy and sentence transformer models."""
        print("Loading spaCy model...")
        self.nlp = spacy.load("en_core_web_sm")
        
        print("Loading sentence transformer model...")
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Common skill categories for normalization
        self.skill_categories = {
            'programming_languages': ['python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'go', 'rust', 'php', 'swift', 'kotlin'],
            'web_technologies': ['html', 'css', 'react', 'angular', 'vue', 'node.js', 'express', 'django', 'flask', 'spring'],
            'databases': ['sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'oracle'],
            'cloud_platforms': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform'],
            'tools': ['git', 'jenkins', 'jira', 'confluence', 'linux', 'unix'],
            'methodologies': ['agile', 'scrum', 'kanban', 'tdd', 'bdd', 'devops']
        }
        
        # Flatten skill categories for easy lookup
        self.all_skills = set()
        for category, skills in self.skill_categories.items():
            self.all_skills.update(skills)
    
    def extract_skills_from_jd(self, jd_text: str) -> List[str]:
        """
        Extract skills from job description using NLP techniques.
        
        Args:
            jd_text (str): Job description text
            
        Returns:
            List[str]: List of extracted skills
        """
        # Process text with spaCy
        doc = self.nlp(jd_text.lower())
        
        # Extract noun chunks and named entities
        skills = set()
        
        # Extract noun chunks (often contain skills)
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.strip()
            # Filter out common stop words and keep meaningful phrases
            # Also filter out very generic terms
            if (len(chunk_text) > 2 and 
                not all(token.is_stop for token in chunk) and
                not self._is_generic_phrase(chunk_text)):
                skills.add(chunk_text)
        
        # Extract named entities that might be technologies or skills
        for ent in doc.ents:
            if ent.label_ in ['PRODUCT', 'ORG', 'TECH'] or len(ent.text) > 2:
                entity_text = ent.text.lower().strip()
                if not self._is_generic_phrase(entity_text):
                    skills.add(entity_text)
        
        # Also look for explicit skill mentions using pattern matching
        skill_patterns = [
            r'\b(experience with|knowledge of|proficient in|skilled in|expert in|familiar with)\s+([^,\.;]+)',
            r'\b(\w+(?:\s+\w+)*)\s*(?:experience|knowledge|skills?|proficiency)',
            r'\b(\w+(?:\s+\w+)*)\s*(?:developer|engineer|specialist|expert)'
        ]
        
        for pattern in skill_patterns:
            matches = re.findall(pattern, jd_text.lower())
            for match in matches:
                if isinstance(match, tuple):
                    # Take the last group which is likely the skill
                    skill = match[-1].strip()
                else:
                    skill = match.strip()
                if (len(skill) > 2 and 
                    len(skill.split()) <= 3 and  # Reasonable skill length
                    not self._is_generic_phrase(skill)):
                    skills.add(skill)
        
        # Normalize skills using our skill categories
        normalized_skills = self._normalize_skills(list(skills))
        
        return list(normalized_skills)
    
    def _is_generic_phrase(self, phrase: str) -> bool:
        """
        Check if a phrase is too generic to be considered a skill.
        
        Args:
            phrase (str): Phrase to check
            
        Returns:
            bool: True if phrase is too generic
        """
        phrase_lower = phrase.lower().strip()
        
        # List of generic terms to filter out
        generic_terms = {
            'a plus', 'ability', 'experience', 'knowledge', 'skills', ' skill', 
            'strong', 'solving', 'problem', 'team', 'environment', 'ability to',
            'years', 'year', 'experience with', 'knowledge of', 'proficient in',
            'skilled in', 'expert in', 'familiar with', 'experience in',
            'strong knowledge', 'strong experience', 'solid experience',
            'solid knowledge', 'good knowledge', 'good experience',
            'team environment', 'team player', 'fast paced', 'fast-paced',
            'detail oriented', 'detail-oriented', 'self motivated', 'self-motivated',
            'hard working', 'hard-working', 'team player', 'communication skills',
            'written communication', 'verbal communication', 'interpersonal skills',
            'analytical skills', 'problem solving', 'time management',
            'organizational skills', 'leadership skills', 'management skills'
        }
        
        # Check if the phrase is in our generic terms
        if phrase_lower in generic_terms:
            return True
        
        # Check if it's mostly made of generic words
        words = phrase_lower.split()
        if len(words) == 0:
            return True
            
        # Common weak words that don't make good skills by themselves
        weak_words = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                     'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                     'should', 'may', 'might', 'must', 'can', 'very', 'quite', 'rather',
                     'quite', 'very', 'really', 'quite', 'somewhat', 'somewhat'}
        
        # If all words are weak words, it's generic
        if all(word in weak_words for word in words):
            return True
            
        # If it's just a single common word, it's probably not a skill
        if len(words) == 1 and words[0] in weak_words:
            return True
            
        return False
    
    def _normalize_skills(self, skills: List[str]) -> Set[str]:
        """
        Normalize extracted skills to standard forms, only keeping known skills.
        
        Args:
            skills (List[str]): Raw extracted skills
            
        Returns:
            Set[str]: Normalized skills (only known skills from our categories)
        """
        normalized = set()
        
        for skill in skills:
            skill_lower = skill.lower().strip()
            
            # Direct match
            if skill_lower in self.all_skills:
                normalized.add(skill_lower)
                continue
                
            # Check if skill contains any known skill
            for category, known_skills in self.skill_categories.items():
                for known_skill in known_skills:
                    if known_skill in skill_lower or skill_lower in known_skill:
                        normalized.add(known_skill)
                        break
                else:
                    continue
                break
        
        return normalized
    
    def generate_interview_questions(self, skills: List[str]) -> Dict[str, List[str]]:
        """
        Generate interview questions based on extracted skills.
        
        Args:
            skills (List[str]): List of extracted skills
            
        Returns:
            Dict[str, List[str]]: Dictionary mapping skill categories to questions
        """
        questions = {
            'technical': [],
            'behavioral': [],
            'situational': []
        }
        
        # Technical question templates - avoid generic ones
        tech_templates = [
            "Can you explain your experience with {skill} and how you've applied it in projects?",
            "How would you approach solving a problem using {skill}?",
            "What are the key considerations when working with {skill}?",
            "Describe a challenging project where you used {skill} and what you learned.",
            "How do you stay updated with the latest developments in {skill}?"
        ]
        
        # Behavioral question templates
        behavioral_templates = [
            "Tell me about a time when you had to learn {skill} quickly for a project.",
            "Describe a situation where you had to troubleshoot an issue with {skill}.",
            "How do you prioritize learning new {skill} when starting a new project?",
            "Give an example of how you've improved your {skill} skills over time.",
            "How do you handle situations where you're unsure about the best {skill} approach?"
        ]
        
        # Situational question templates
        situational_templates = [
            "How would you explain {skill} to a non-technical stakeholder?",
            "If you had to choose between two {skill} approaches, how would you decide?",
            "What would you do if a project requirement changed mid-way requiring different {skill} expertise?",
            "How do you ensure quality when working with {skill} under tight deadlines?",
            "Describe how you would mentor a junior developer learning {skill}."
        ]
        
        # Filter out skills that are too generic or not meaningful for question generation
        meaningful_skills = [skill for skill in skills if len(skill) > 2 and 
                           not any(generic in skill for generic in ['years', 'year', 'experience', 'ability', 'skill', 'knowledge', 'plus', 'strong', 'solid', 'good', 'a', 'an', 'the'])]
        
        # Generate questions for each meaningful skill
        for skill in meaningful_skills:
            # Technical questions - limit to 2 per skill to avoid too many questions
            for template in tech_templates[:2]:
                questions['technical'].append(template.format(skill=skill))
            
            # Behavioral questions - assign to some skills
            if hash(skill) % 3 == 0:  # Every third skill gets behavioral questions
                for template in behavioral_templates[:1]:
                    questions['behavioral'].append(template.format(skill=skill))
            
            # Situational questions - assign to some skills
            if hash(skill) % 5 == 0:  # Every fifth skill gets situational questions
                for template in situational_templates[:1]:
                    questions['situational'].append(template.format(skill=skill))
        
        # Limit total questions to avoid overwhelming (but ensure we have some)
        for category in questions:
            if len(questions[category]) > 8:
                questions[category] = questions[category][:8]
            # If we have no questions in a category, add some generic ones
            elif len(questions[category]) == 0 and meaningful_skills:
                if category == 'technical':
                    questions[category] = [f"Can you discuss your experience with {meaningful_skills[0]}?"]
                elif category == 'behavioral':
                    questions[category] = [f"Tell me about a time you had to learn {meaningful_skills[0]} quickly."]
                elif category == 'situational':
                    questions[category] = [f"How would you explain {meaningful_skills[0]} to a non-technical stakeholder?"]
        
        return questions
    
    def process_jd(self, jd_text: str) -> Dict:
        """
        Main method to process job description and generate interview questions.
        
        Args:
            jd_text (str): Job description text
            
        Returns:
            Dict: Contains extracted skills and generated questions
        """
        print("Extracting skills from job description...")
        skills = self.extract_skills_from_jd(jd_text)
        
        print(f"Found {len(skills)} skills: {', '.join(sorted(skills))}")
        
        print("Generating interview questions...")
        questions = self.generate_interview_questions(skills)
        
        return {
            'skills': sorted(list(skills)),
            'questions': questions
        }

def main():
    parser = argparse.ArgumentParser(description='JD-Resume Interview Agent')
    parser.add_argument('--jd', type=str, help='Job description text or file path')
    parser.add_argument('--file', '-f', action='store_true', help='Indicate that --jd is a file path')
    parser.add_argument('--output', '-o', type=str, help='Output file for results (JSON format)')
    
    args = parser.parse_args()
    
    # Initialize agent
    agent = JDResumeAgent()
    
    # Get job description text
    if args.file:
        try:
            with open(args.jd, 'r', encoding='utf-8') as f:
                jd_text = f.read()
        except FileNotFoundError:
            print(f"Error: File '{args.jd}' not found.")
            return
        except Exception as e:
            print(f"Error reading file: {e}")
            return
    elif args.jd:
        jd_text = args.jd
    else:
        # Interactive mode
        print("Enter job description (press Ctrl+D or Ctrl+Z when finished):")
        jd_text = ""
        try:
            while True:
                line = input()
                jd_text += line + "\n"
        except EOFError:
            pass
    
    if not jd_text.strip():
        print("Error: No job description provided.")
        return
    
    # Process JD
    result = agent.process_jd(jd_text)
    
    # Output results
    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
            print(f"Results saved to {args.output}")
        except Exception as e:
            print(f"Error writing to file: {e}")
    else:
        print("\n" + "="*50)
        print("EXTRACTED SKILLS")
        print("="*50)
        for skill in result['skills']:
            print(f"• {skill}")
        
        print("\n" + "="*50)
        print("GENERATED INTERVIEW QUESTIONS")
        print("="*50)
        
        for category, questions in result['questions'].items():
            if questions:
                print(f"\n{category.upper()} QUESTIONS:")
                for i, q in enumerate(questions, 1):
                    print(f"{i}. {q}")

if __name__ == "__main__":
    main()
