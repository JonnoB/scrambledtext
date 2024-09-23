# scrambledtext

A Python library for creating synthetic corrupted OCR text using a Markov process.

# Overview

The `scrambledtext` library provides tools to simulate OCR (Optical Character Recognition) errors in text by applying various types of corruptions such as character substitutions, deletions, and insertions. It uses probability distributions derived from aligned text pairs to generate realistic OCR-like errors. It is simple, containing only a handful of functions and classes, and is dependent only on libraries that come with the base installation of Python.

The library was developed for the paper "Scrambled text: training Language Models to correct OCR errors using synthetic data" The code for this paper can be found at [scrambledtext_analysis](https://github.com/JonnoB/scrambledtext_analysis) and [training_lms_with_synthetic_data](https://github.com/JonnoB/training_lms_with_synthetic_data)

The library uses character-aligned parallel texts to create the conditional 1-gram transmission distributions for a Markov process. The created character transfer probabilities are 

- **Correct**: The character passes through the network unchanged
- **Substitute**: The character is substituted using the probability distribution specific to that character. If that character is not present in the conditional probability table, the mean transmission probability for the training set is used
- **Insert**: A new character is inserted after the character currently passing through the process. The new character has a probability distribution specific to the character in the process. If that character is not present in the conditional probability table, the mean transmission probability for the training set is used
- **Delete**: The character is deleted

A diagram of the Markov process/network is shown below

![corruption_network](https://github.com/user-attachments/assets/257d6e57-08dd-447d-8fd9-c86b7a2cde8f)


The transmission probabilities are learnt from line-aligned text. A valuable library for creating line-aligned text is [genelog](https://github.com/microsoft/genalog). Although `scrambledtext` was developed for English, the library is language and script agnostic, as the functions are a general Markov process that acts as a simple simulator for the kind of errors observed in OCR.

The library was developed for the paper "Scrambled text: training Language Models to correct OCR errors using synthetic data", The code for this paper can be found at [scrambledtext_analysis](https://github.com/JonnoB/scrambledtext_analysis) and [training_lms_with_synthetic_data](https://github.com/JonnoB/training_lms_with_synthetic_data)


# Features

- Calculate probability distributions for text alignment errors
- Generate synthetic corrupted text based on learned error patterns
- Adjust corruption levels to target specific Word Error Rates (WER) and Character Error Rates (CER)
- Save and load probability distributions to/from JSON files

# Installation

TO DO

# Usage

There are two main ways to use the library, generating the corruption tables or corrupting text. See below for examples


## Calculating Probability Distributions

```
# Prepare aligned text pairs (ground truth and noisy text)
aligned_texts = [
    ("This is a sample.", "Th1s iz a sampl3."),
    ("Another example.", "An0ther examp1e"),
    # ... more aligned text pairs ...
]

# Calculate probability distributions
probs = ProbabilityDistributions(aligned_texts)

# Save the calculated probabilities to a JSON file
probs.save_to_json('probabilities.json')

```

## Corrupting Text

```

from scrambledtext import ProbabilityDistributions, CorruptionEngine

# Load pre-calculated probability distributions
probs = ProbabilityDistributions.load_from_json('probabilities.json')

# Create a corruption engine
engine = CorruptionEngine(
    probs.conditional,
    probs.substitutions,
    probs.insertions,
    target_wer=0.2,
    target_cer=0.1
)

# Corrupt some text
original_text = "This is a sample text to be corrupted."
corrupted_text, actual_wer, actual_cer, effective_cer = engine.corrupt_text(original_text)

print(f"Original: {original_text}")
print(f"Corrupted: {corrupted_text}")
print(f"Actual WER: {actual_wer:.2f}")
print(f"Actual CER: {actual_cer:.2f}")
print(f"Effective CER: {effective_cer:.2f}")

```


# License

This project is licensed under the MIT License - see the LICENSE file for details.

# Citing this repo
If this repo is helpful in your own work, please cite xxx paper still in progress. No citation information yet xxx

# To Do
- make actual libarary
- install instructions
- Create aligned demo text
