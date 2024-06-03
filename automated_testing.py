import pandas as pd
import random
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import spacy
import stanza
import re
from nltk.metrics import precision, recall, f_measure
import json
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

# Load data
companies = pd.read_csv('datasets/companies_clean.csv')
airline = pd.read_csv('datasets/airline_clean.csv')
chicago = pd.read_csv('datasets/chicago_clean.csv')

# Rename columns for consistency
airline.columns = ['first_name', 'last_name', 'pilot_name', 'nationality']
companies.columns = ['company', 'city']
chicago.columns = ['job_title', 'department']

# Helper functions to generate random values
def get_random_value(df, column):
    return random.choice(df[column].unique().tolist())

def get_random_date():
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    random_date = start_date + timedelta(days=random.randint(0, (end_date - start_date).days))
    return random_date.strftime("%d-%m-%Y")

def get_random_time():
    hour = random.randint(0, 23)
    minute = random.randint(0, 59)
    return f"{hour:02d}:{minute:02d}"

def get_random_amount():
    return f"{random.randint(100, 15000)}.00"

def get_random_flight_number():
    return f"{random.randint(100, 999)}"

def generate_template_data(template_type):
    if template_type == 'airline_ticket':
        return generate_airline_ticket_data()
    elif template_type == 'invoice':
        return generate_invoice_data()
    elif template_type == 'recommendation_letter':
        return generate_recommendation_letter_data()
    else:
        raise ValueError("Unknown template type")

# Functions to generate data for each template
def generate_airline_ticket_data():
    return {
        'first_name': get_random_value(airline, 'first_name'),
        'last_name': get_random_value(airline, 'last_name'),
        'pilot_name': get_random_value(airline, 'pilot_name'),
        'nationality': get_random_value(airline, 'nationality'),
        'flight_number': get_random_flight_number(),
        'departure_city': get_random_value(companies, 'city'),
        'destination_city': get_random_value(companies, 'city'),
        'date': get_random_date(),
        'time': get_random_time(),
    }

def generate_invoice_data():
    return {
        'company': get_random_value(companies, 'company'),
        'city': get_random_value(companies, 'city'),
        'first_name': get_random_value(airline, 'first_name'),
        'last_name': get_random_value(airline, 'last_name'),
        'job_title': get_random_value(chicago, 'job_title'),
        'department': get_random_value(chicago, 'department'),
        'invoice_number': get_random_flight_number(),  
        'date': get_random_date(),
        'amount': get_random_amount(),
    }

def generate_recommendation_letter_data():
    return {
        'first_name': get_random_value(airline, 'first_name'),
        'last_name': get_random_value(airline, 'last_name'),
        'job_title': get_random_value(chicago, 'job_title'),
        'company': get_random_value(companies, 'company'),
        'recommender': get_random_value(airline, 'pilot_name'),  
    }

class AnonymizationStrategy(ABC):
    @abstractmethod
    def modify_text(self, text):
        pass

    def normalize_spaced_text(self, text):
        text = re.sub(r'\s{2,}', '||', text)
        text = re.sub(r'(?<=\b\w) (?=\w\b)', '', text)
        text = re.sub(r'\|\|', ' ', text)
        return text

    def contains_one_letter_space_one_letter(self, text):
        pattern = r'\b\w\s\w\b'
        return re.search(pattern, text) is not None

class SpacyAnonymization(AnonymizationStrategy):
    def __init__(self):
        self.nlp = spacy.load('en_core_web_lg')

    def modify_text(self, text):
        # Normalize newline characters to spaces and trim extra spaces
        text = text.replace('\n', ' ').strip()
        text = ' '.join(text.split())

        if self.contains_one_letter_space_one_letter(text):
            text = self.normalize_spaced_text(text)
        
        doc = self.nlp(text)
        entities = doc.ents
        sorted_entities = sorted(entities, key=lambda e: e.start_char, reverse=True)
        anonymized_text = text
        entities_dict = {}

        for entity in sorted_entities:
            entity_text = entity.text.strip()  # Trim whitespace from entity text
            entity_type = entity.label_
            if entity_type == 'CARDINAL':
                entity_type = "NUMBER"
            entities_dict[entity_text] = "{{" + entity_type + "}}"
            anonymized_text = anonymized_text[:entity.start_char] + "{{" + entity_type + "}}" + anonymized_text[entity.end_char:]

        return anonymized_text, entities_dict

class StanzaAnonymization(AnonymizationStrategy):
    def __init__(self):
        self.nlp = stanza.Pipeline(lang='en', processors='tokenize,ner')

    def modify_text(self, text):
        doc = self.nlp(text)
        entities = doc.entities
        sorted_entities = sorted(entities, key=lambda e: e.start_char, reverse=True)
        anonymized_text = text
        entities_dict = {}

        for entity in sorted_entities:
            entity_text = entity.text.strip()  # Trim whitespace from entity text
            entity_type = entity.type
            if entity_type == 'CARDINAL':
                entity_type = "NUMBER"
            entities_dict[entity_text] = "{{" + entity_type + "}}"
            anonymized_text = anonymized_text[:entity.start_char] + "{{" + entity_type + "}}" + anonymized_text[entity.end_char:]

        return anonymized_text, entities_dict

class TemplateSupplier:
    def __init__(self):
        self.templates = {
            'airline_ticket': self.airline_ticket_template(),
            'invoice': self.invoice_template(),
            'recommendation_letter': self.recommendation_letter_template()
        }

    def get_template(self, template_type):
        return self.templates[template_type]

    def populate_template(self, template, data):
        for key, value in data.items():
            template = template.replace(f'{{{{ {key} }}}}', value)
        return template

    @staticmethod
    def airline_ticket_template():
        return """
    Airline Ticket
    ==============

    Passenger Information:
    -----------------------
    Passenger Name: {{ first_name }}
    Passenger Surname: {{ last_name }}
    Pilot Name: {{ pilot_name }}
    Nationality: {{ nationality }}

    Flight Information:
    -------------------
    Flight Number: {{ flight_number }}
    Departure city: {{ departure_city }}
    Destination: {{ destination_city }}
    Date: {{ date }}
    Departure at: {{ time }}
        """

    @staticmethod
    def invoice_template():
        return """
Invoice
=======

Company Information:
---------------------
Company: {{ company }}
City: {{ city }}

Billing Information:
--------------------
Customer Name: {{ first_name }} {{ last_name }}
Job Title: {{ job_title }}
Department: {{ department }}

Invoice Details:
----------------
Invoice Number: {{ invoice_number }}
Date: {{ date }}
Total Amount: ${{ amount }}
        """

    @staticmethod
    def recommendation_letter_template():
        return """
Letter of Recommendation
=========================

To Whom It May Concern,

I am writing to recommend {{ last_name }} for the position in your company. {{ first_name }} has been working as a {{ job_title }} in the {{ company }} and has shown exceptional skills and dedication.

{{ first_name }} is a highly motivated individual with a strong work ethic and a positive attitude. I am confident that {{ first_name }} will be an asset to your team.

Please feel free to contact me if you need any further information.

Sincerely,
{{ recommender }}
        """

# Expected labels for each column
expected_labels = {
    'first_name': '{{PERSON}}',
    'last_name': '{{PERSON}}',
    'pilot_name': '{{PERSON}}',
    'nationality': '{{GPE}}',
    'job_title': '',
    'department': '',
    'city': '{{GPE}}',
    'company': '{{ORG}}',
    'flight_number': '{{NUMBER}}',
    'date': '{{DATE}}',
    'time': '{{TIME}}',
    'amount': '{{MONEY}}',
    'destination': '{{GPE}}',
    'departure_city': '{{GPE}}',
    'destination_city': '{{GPE}}',
    'invoice_number': '{{NUMBER}}',
    'recommender': '{{PERSON}}'
}

def map_template_data_to_expected_format(template_data):
    mapped_data = {}
    for key, value in template_data.items():
        expected_label = expected_labels.get(key)
        if expected_label:
            mapped_data[value] = expected_label
    return mapped_data

def evaluate_anonymization(anonymized_text, original_data, entities_dict):
    mapped_data = map_template_data_to_expected_format(original_data)
    true_labels = []
    predicted_labels = []

    for value, true_label in mapped_data.items():
        true_labels.append(true_label)
        predicted_label = entities_dict.get(value, 'O')
        predicted_labels.append(predicted_label)

    precision_score = precision(set(true_labels), set(predicted_labels))
    recall_score = recall(set(true_labels), set(predicted_labels))
    f1_score = f_measure(set(true_labels), set(predicted_labels))

    return precision_score, recall_score, f1_score

# def main():
#     template_supplier = TemplateSupplier()
#     template_type = 'recommendation_letter'
#     # template_type = 'invoice'
#     # template_type = 'airline_ticket'
#     spacy_anonymizer = SpacyAnonymization()
#     stanza_anonymizer = StanzaAnonymization()

#     spacy_precision_scores = []
#     spacy_recall_scores = []
#     spacy_f1_scores = []

#     stanza_precision_scores = []
#     stanza_recall_scores = []
#     stanza_f1_scores = []

#     for _ in range(10): 
#         template = template_supplier.get_template(template_type)
#         template_data = generate_template_data(template_type)
#         populated_template = template_supplier.populate_template(template, template_data)

#         spacy_anonymized_text, spacy_entities_dict = spacy_anonymizer.modify_text(populated_template)
#         stanza_anonymized_text, stanza_entities_dict = stanza_anonymizer.modify_text(populated_template)

#         spacy_precision, spacy_recall, spacy_f1 = evaluate_anonymization(spacy_anonymized_text, template_data, spacy_entities_dict)
#         stanza_precision, stanza_recall, stanza_f1 = evaluate_anonymization(stanza_anonymized_text, template_data, stanza_entities_dict)

#         spacy_precision_scores.append(spacy_precision)
#         spacy_recall_scores.append(spacy_recall)
#         spacy_f1_scores.append(spacy_f1)

#         stanza_precision_scores.append(stanza_precision)
#         stanza_recall_scores.append(stanza_recall)
#         stanza_f1_scores.append(stanza_f1)

#     print(f"Spacy Anonymization Results (averaged over 10 examples):")
#     print(f"Precision: {sum(spacy_precision_scores)/10}")
#     print(f"Recall: {sum(spacy_recall_scores)/10}")
#     print(f"F1 Score: {sum(spacy_f1_scores)/10}\n")

#     print(f"Stanza Anonymization Results (averaged over 10 examples):")
#     print(f"Precision: {sum(stanza_precision_scores)/10}")
#     print(f"Recall: {sum(stanza_recall_scores)/10}")
#     print(f"F1 Score: {sum(stanza_f1_scores)/10}\n")

def run_anonymization_trials(trials, template_type):
    template_supplier = TemplateSupplier()
    spacy_anonymizer = SpacyAnonymization()
    stanza_anonymizer = StanzaAnonymization()

    spacy_precision_scores = []
    spacy_recall_scores = []
    spacy_f1_scores = []

    stanza_precision_scores = []
    stanza_recall_scores = []
    stanza_f1_scores = []

    for _ in range(trials): 
        template = template_supplier.get_template(template_type)
        template_data = generate_template_data(template_type)
        populated_template = template_supplier.populate_template(template, template_data)

        spacy_anonymized_text, spacy_entities_dict = spacy_anonymizer.modify_text(populated_template)
        stanza_anonymized_text, stanza_entities_dict = stanza_anonymizer.modify_text(populated_template)

        spacy_precision, spacy_recall, spacy_f1 = evaluate_anonymization(spacy_anonymized_text, template_data, spacy_entities_dict)
        stanza_precision, stanza_recall, stanza_f1 = evaluate_anonymization(stanza_anonymized_text, template_data, stanza_entities_dict)

        spacy_precision_scores.append(spacy_precision)
        spacy_recall_scores.append(spacy_recall)
        spacy_f1_scores.append(spacy_f1)

        stanza_precision_scores.append(stanza_precision)
        stanza_recall_scores.append(stanza_recall)
        stanza_f1_scores.append(stanza_f1)

    return (spacy_precision_scores, spacy_recall_scores, spacy_f1_scores), (stanza_precision_scores, stanza_recall_scores, stanza_f1_scores)

def plot_results(spacy_scores, stanza_scores, template_type, ax):
    metrics = ['Precision', 'Recall']
    spacy_avg_scores = [sum(scores) / len(scores) for scores in spacy_scores[:2]]
    stanza_avg_scores = [sum(scores) / len(scores) for scores in stanza_scores[:2]]

    x = range(len(metrics))

    width = 0.4

    spacy_bars = ax.bar(x, spacy_avg_scores, width=width, label='Spacy', align='center')
    stanza_bars = ax.bar([i + width for i in x], stanza_avg_scores, width=width, label='Stanza', align='center')

    # Adding scores on top of the bars
    for bar in spacy_bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom')

    for bar in stanza_bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom')

    ax.set_xlabel('Metrics')
    ax.set_ylabel('Scores')
    ax.set_title(f'Comparison of Anonymization Strategies for {template_type.replace("_", " ").title()}')
    ax.set_xticks([i + width / 2 for i in x])
    ax.set_xticklabels(metrics)
    ax.legend()


def main():
    trials = 1000
    template_types = ['airline_ticket', 'invoice', 'recommendation_letter']

    results = {}

    for template_type in template_types:
        spacy_scores, stanza_scores = run_anonymization_trials(trials, template_type)
        results[template_type] = (spacy_scores, stanza_scores)

    fig, axes = plt.subplots(3, 1, figsize=(10, 18))

    for ax, template_type in zip(axes, template_types):
        spacy_scores, stanza_scores = results[template_type]
        plot_results(spacy_scores, stanza_scores, template_type, ax)

    plt.tight_layout()
    plt.savefig('anonymization_comparison.pdf')
    # plt.savefig('anonymization_comparison.jpg')
    plt.show()

if __name__ == '__main__':
    main()