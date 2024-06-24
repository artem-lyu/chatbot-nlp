# Chatbot-NLP

This repository contains a simple NLP-based chatbot implemented in Python using TensorFlow and NLTK. The chatbot is trained on a custom intents JSON file, which includes different patterns and responses for various tags.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/artem-lyu/chatbot-nlp.git
    cd chatbot-nlp
    ```

2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Prepare your intents JSON file with patterns and responses. Place it in the `chatbot-nlp` directory.

2. Run the chatbot script:

    ```bash
    python main.py
    ```

3. Interact with the chatbot through the terminal. Type `0` to stop the conversation.

## Code Overview

### Importing Libraries

```python
import json
import string
import random
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
```


