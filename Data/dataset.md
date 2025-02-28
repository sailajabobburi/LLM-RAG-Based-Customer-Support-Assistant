# Bitext - Customer Service Tagged Training Dataset for LLM-based Virtual Assistants

## Dataset 

## Overview
This dataset is designed for training **LLM-based virtual assistants** in customer service interactions. It includes a variety of user requests, categorized by **intent and response**, along with **language variation tags**
Each entry in the dataset contains the following fields:


## Source of the Dataset
This dataset is provided by **Bitext Innovations** and is publicly available at:

- **Hugging Face**: [Bitext Customer Service Dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)

Please refer to the official dataset sources for **detailed documentation and licensing information**.

## Structure
- flags: tags (explained below in the Language Generation Tags section)
- instruction: a user request from the Customer Service
- category: the high-level semantic category for the intent
- intent: the intent corresponding to the user instruction
- response: an example expected response from the virtual assistant

## Categories and Intents
The dataset covers multiple **customer service categories**, including:
- **Account Management**: create, delete, edit accounts, recover passwords.
- **Order Processing**: cancel, change, track orders.
- **Refunds & Payments**: check policies, get refunds, resolve payment issues.
- **Shipping & Delivery**: address changes, delivery times.
- **Customer Support**: contact agents, submit complaints.

## Language Variations
The dataset includes **linguistic tags** to help train chatbots for **diverse user interactions**, such as:
- **Politeness (P)**: "Could you please help me?"
- **Colloquial (Q)**: "Can u cancel my order?"
- **Errors & Typos (Z)**: "how can i activaet my card?"

## Entities in the Dataset
The dataset includes **predefined placeholders (entities)** that appear in customer queries. Few examples:

| **Entity**                   | **Usage** |
|------------------------------|----------|
| `{{Order Number}}`           | Used in order-related intents (cancel, change, track order). |
| `{{Invoice Number}}`         | Present in invoice-related intents. |
| `{{Customer Support Email}}` | Customer service and support interactions. |
| `{{Live Chat Support}}` | Used when customers want to speak with an agent. |
| `{{Website URL}}` | Found in intents related to payments, refunds, and support. |
| `{{Shipping Cut-off Time}}` | Appears in delivery-related intents. |
| `{{Delivery City}}`, `{{Delivery Country}}` | Used in delivery options. |
| `{{Money Amount}}`, `{{Refund Amount}}` | Appears in refund-related queries. |

These placeholders help train **Named Entity Recognition (NER) models** to extract key details from customer requests.


