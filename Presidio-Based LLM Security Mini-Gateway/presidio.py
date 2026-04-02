from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern, EntityRecognizer
from presidio_anonymizer import AnonymizerEngine
import re

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

employee_pattern = Pattern("EMP-ID", r"\bEMP-\d{3,5}\b", 0.85)
employee_recognizer = PatternRecognizer(supported_entity="EMPLOYEE_ID",
                                       patterns=[employee_pattern],
                                       name="EmployeeIDRecognizer")
analyzer.registry.add_recognizer(employee_recognizer)

api_pattern = Pattern("API-KEY", r"\bsk-[A-Za-z0-9]{8,}\b", 0.85)
api_recognizer = PatternRecognizer(supported_entity="API_KEY",
                                   patterns=[api_pattern],
                                   name="APIKeyRecognizer")
analyzer.registry.add_recognizer(api_recognizer)

def detect_composite(text):

    results = analyzer.analyze(text=text, language="en")
    types = [r.entity_type for r in results]
    composite_found = "EMPLOYEE_ID" in types and "EMAIL_ADDRESS" in types
    return composite_found, results

def detect_pii(text):
    return analyzer.analyze(text=text, language="en")

def anonymize_pii(text):
    results = analyzer.analyze(text=text, language="en")
    return anonymizer.anonymize(text=text, analyzer_results=results).text