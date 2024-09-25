
"""
scrambledtext

This library provides tools for simulating text corruption, particularly useful for testing OCR systems or generating synthetic data for machine learning tasks.

Key Components:
1. ProbabilityDistributions: A class for calculating and managing probability distributions of text alignment errors.
2. CorruptionEngine: A class for corrupting entire texts based on target Word Error Rate (WER) and Character Error Rate (CER).

The library includes functions for loading example distributions, modifying probability distributions, and various utility functions for text processing.

Usage:
1. Load or create probability distributions using ProbabilityDistributions class.
2. Initialize a CorruptionEngine with the distributions and desired error rates.
3. Use the CorruptionEngine to corrupt text samples.

Example:
    distribs = load_example_distribs()
    engine = CorruptionEngine(
        distribs['conditional'],
        distribs['substitutions'],
        distribs['insertions'],
        target_wer=0.2,
        target_cer=0.05
    )
    corrupted_text, actual_wer, actual_cer, effective_cer = engine.corrupt_text("Your sample text here")

For detailed usage of individual components, refer to their respective class and method docstrings.
"""
import collections
import re
import random
import math
import json
from pkg_resources import resource_filename


def load_example_distribs():
    """ 
    Load the default corruption distributions. These distributions are based on english text from archival newspapers. Specifically the BLN600 and Overproof collections.
    See library readme for details.
    """
    json_path = resource_filename('my_library', 'data.json')
    with open(json_path, 'r') as f:
        return json.load(f)


def modify_and_renormalize_probs(conditional_probs, column, desired_value):
    """
    Modify a specific column in the conditional probabilities to the desired value,
    ensuring probabilities remain within [0, 1], and then renormalize so they sum to 1.

    Parameters:
    - conditional_probs: The dictionary of conditional probabilities to modify.
    - column: The column ('correct', 'substitute', 'delete', 'insert') to modify.
    - desired_value: The desired value for the selected column.

    Returns:
    - modified_probs: A new dictionary with the modified and renormalized probabilities.
    """
    modified_probs = {}

    for char, probs in conditional_probs.items():
        # Set the selected column to the desired value
        scaled_value = max(0, min(1, desired_value))

        # Calculate the remaining total for the other columns
        remaining_total = 1 - scaled_value

        # Calculate the total of the other columns before scaling
        original_remaining_total = sum(probs[key] for key in probs if key != column)

        # Renormalize the other columns
        modified_probs[char] = {}
        for key in probs:
            if key == column:
                modified_probs[char][key] = scaled_value
            else:
                if original_remaining_total > 0:
                    new_value = probs[key] * remaining_total / original_remaining_total
                else:
                    new_value = 0  # Handle edge case where original_remaining_total might be zero

                # Ensure renormalized value is within [0, 1]
                modified_probs[char][key] = max(0, min(1, new_value))

        # Final adjustment to ensure all probabilities sum to 1
        total_prob = sum(modified_probs[char].values())
        if total_prob != 1:
            for key in modified_probs[char]:
                modified_probs[char][key] = modified_probs[char][key] / total_prob

    return modified_probs


class ProbabilityDistributions:
    """
    A class to calculate and manage probability distributions for text alignment errors,
    such as deletions, insertions, and substitutions, between ground truth text and noisy text.

    This class processes aligned text pairs, calculates various probability distributions,
    and provides methods to save and load these distributions to and from JSON files.

    Attributes
    ----------
    deletion_counts : defaultdict
        Counts of deletions observed for each character in the ground truth text.
    insertion_counts : defaultdict of defaultdicts
        Counts of insertions observed after each character in the ground truth text.
    substitution_counts : defaultdict of defaultdicts
        Counts of substitutions observed for each character in the ground truth text.
    character_counts : defaultdict
        Counts of occurrences of each character in the ground truth text.
    character_distribution : dict
        Probability distribution of characters based on their occurrence in the ground truth text.
    conditional : dict
        Conditional probabilities of correct matches, substitutions, deletions, and insertions for each character.
    substitutions : dict
        Conditional probability distribution of substitutions for each character.
    insertions : dict
        Conditional probability distribution of insertions for each character.

    Methods
    -------
    __init__(self, aligned_texts=None)
        Initializes the class and processes aligned text pairs if provided.
    initialize_counters(self)
        Initializes the counters for deletions, insertions, substitutions, and character occurrences.
    update_counts(self, gt, noise)
        Updates the counts for deletions, insertions, and substitutions based on aligned text pairs.
    calculate_character_distribution(self)
        Calculates the distribution of characters based on their counts in the ground truth text.
    calculate_conditional_probs(self)
        Calculates the conditional probabilities for each character.
    generate_substitution_insertion_tables(self)
        Generates the substitution and insertion tables based on observed counts.
    add_default_values(self)
        Adds default probability values for characters not explicitly listed in the tables.
    save_to_json(self, file_path)
        Saves the current state of character distribution, conditional probabilities, substitutions,
        and insertions to a JSON file.
    load_from_json(cls, file_path)
        Loads character distribution, conditional probabilities, substitutions, and insertions
        from a JSON file and returns an instance of ProbabilityDistributions.
    modify_and_renormalize_probs(self, column, desired_value, inplace=True)
        Modifies a specific column in the conditional probabilities to a desired value and renormalizes
        the probabilities.
    calculate_joint_probabilities(self)
        Calculates the joint probabilities by multiplying conditional probabilities by the character distribution
        and summing these joint probabilities.
    """

    def __init__(self, aligned_texts=None):
        # Initialize counters
        (
            self.deletion_counts,
            self.insertion_counts,
            self.substitution_counts,
            self.character_counts,
        ) = self.initialize_counters()

        if aligned_texts:
            # Update counts for all aligned text pairs
            for gt, noise in aligned_texts:
                self.update_counts(gt, noise)

            # After updating counts, calculate the character distribution and conditional probabilities
            self.character_distribution = self.calculate_character_distribution()
            self.conditional = self.calculate_conditional_probs()
            self.substitutions, self.insertions = (
                self.generate_substitution_insertion_tables()
            )

            # Add default values to the probability tables
            self.add_default_values()

    def initialize_counters(self):
        deletion_counts = collections.defaultdict(int)
        insertion_counts = collections.defaultdict(lambda: collections.defaultdict(int))
        substitution_counts = collections.defaultdict(
            lambda: collections.defaultdict(int)
        )
        character_counts = collections.defaultdict(int)
        return deletion_counts, insertion_counts, substitution_counts, character_counts

    def update_counts(self, gt, noise):
        """
        Update counts for deletions, insertions, and substitutions based on aligned text pairs.
        """
        assert len(gt) == len(noise), "Aligned text pairs must have the same length."

        n = len(gt)
        i, j = 0, 0  # Pointers for gt and noise
        last_gt_char = ""  # Track the last valid character in gt

        while i < n and j < n:
            gt_char = gt[i]
            noise_char = noise[j]

            if gt_char == "@":  # Insertion case
                self.insertion_counts[last_gt_char][noise_char] += 1

            elif noise_char == "@":  # Deletion case
                self.deletion_counts[gt_char] += 1

            elif gt_char != noise_char:  # Substitution case
                self.substitution_counts[gt_char][noise_char] += 1
                self.character_counts[gt_char] += 1  # Count this as a gt occurrence
                last_gt_char = gt_char  # Update last valid character

            else:  # Correct character
                self.character_counts[gt_char] += 1
                last_gt_char = gt_char  # Update last valid character

            # Increment both pointers after processing the current pair
            i += 1
            j += 1

        # Handle any remaining deletions or insertions at the end
        while i < n and gt[i] != "@":
            self.deletion_counts[gt[i]] += 1
            i += 1
        while j < n and noise[j] != "@":
            self.insertion_counts[last_gt_char][noise[j]] += 1
            j += 1

    def calculate_character_distribution(self):
        """
        Calculate the distribution of characters based on their counts.
        """
        total_characters = sum(self.character_counts.values())
        character_distribution = {
            char: count / total_characters
            for char, count in sorted(self.character_counts.items())
        }
        return character_distribution

    def calculate_conditional_probs(self):
        """
        Calculate the conditional probabilities for each character.
        """
        conditional = {}

        for char in sorted(self.character_counts):
            total_count = self.character_counts[char]

            # Calculate individual probabilities for this character
            delete_prob = (
                self.deletion_counts[char] / total_count
                if char in self.deletion_counts
                else 0
            )
            substitute_prob = (
                sum(self.substitution_counts[char].values()) / total_count
                if char in self.substitution_counts
                else 0
            )
            insert_prob = (
                sum(self.insertion_counts[char].values()) / total_count
                if char in self.insertion_counts
                else 0
            )

            # Correct probability is what's left after considering deletions, substitutions, and insertions
            correct_prob = 1 - (delete_prob + substitute_prob + insert_prob)

            # Ensure probabilities are within valid range [0, 1]
            correct_prob = max(0, min(1, correct_prob))

            conditional[char] = {
                "correct": correct_prob,
                "substitute": substitute_prob,
                "delete": delete_prob,
                "insert": insert_prob,
            }

        return conditional

    def generate_substitution_insertion_tables(self):
        """
        Generate the substitution and insertion tables based on observed counts.
        """
        substitutions = {}
        insertions = {}

        for char in sorted(self.substitution_counts):
            total_subs = sum(self.substitution_counts[char].values())
            substitutions[char] = {
                sub_char: count / total_subs
                for sub_char, count in sorted(self.substitution_counts[char].items())
            }

        for char in sorted(self.insertion_counts):
            total_ins = sum(self.insertion_counts[char].values())
            insertions[char] = {
                ins_char: count / total_ins
                for ins_char, count in sorted(self.insertion_counts[char].items())
            }

        return substitutions, insertions

    def add_default_values(self):
        """
        Add default values for characters not explicitly listed.
        """
        default_conditional = {
            "correct": sum(d["correct"] for d in self.conditional.values())
            / len(self.conditional),
            "substitute": sum(d["substitute"] for d in self.conditional.values())
            / len(self.conditional),
            "delete": sum(d["delete"] for d in self.conditional.values())
            / len(self.conditional),
            "insert": sum(d["insert"] for d in self.conditional.values())
            / len(self.conditional),
        }

        default_substitution = {
            char: prob for char, prob in sorted(self.character_distribution.items())
        }

        default_insertion = {
            char: prob for char, prob in sorted(self.character_distribution.items())
        }

        self.conditional["default"] = default_conditional
        self.substitutions["default"] = default_substitution
        self.insertions["default"] = default_insertion

    def modify_and_renormalize_probs(self, column, desired_value, inplace=True):
        """
        Modify a specific column in the class's conditional probabilities to the desired value,
        ensuring probabilities remain within [0, 1], and then renormalize so they sum to 1.

        Parameters:
        - column: The column ('correct', 'substitute', 'delete', 'insert') to modify.
        - desired_value: The desired value for the selected column.
        - inplace: If True, modify the class's conditional attribute directly. If False, return a new modified dictionary.

        Returns:
        - If inplace is False, returns modified_probs: A new dictionary with the modified and renormalized probabilities.
        """
        modified_probs = modify_and_renormalize_probs(
            self.conditional, column, desired_value
        )

        if inplace:
            # Update the class attribute in place
            self.conditional = modified_probs
        else:
            # Return the modified dictionary
            return modified_probs

    def calculate_joint_probabilities(self):
        """
        Calculate the joint probabilities by multiplying conditional probabilities by the character distribution
        and then sum these joint probabilities.
        """
        joint_probs = {"correct": 0.0, "substitute": 0.0, "delete": 0.0, "insert": 0.0}

        # Calculate joint probabilities
        for char, cond_prob in self.conditional.items():
            if char in self.character_distribution:
                char_prob = self.character_distribution[char]
                joint_probs["correct"] += cond_prob["correct"] * char_prob
                joint_probs["substitute"] += cond_prob["substitute"] * char_prob
                joint_probs["delete"] += cond_prob["delete"] * char_prob
                joint_probs["insert"] += cond_prob["insert"] * char_prob

        return joint_probs

    def save_to_json(self, file_path):
        """
        Save the character distribution, conditional probabilities, substitutions, and insertions to a JSON file.

        :param file_path: The path to the JSON file where the data will be saved.
        """
        data = {
            "character_distribution": self.character_distribution,
            "conditional": self.conditional,
            "substitutions": self.substitutions,
            "insertions": self.insertions,
        }
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)

    @classmethod
    def load_from_json(cls, file_path):
        """
        Load the character distribution, conditional probabilities, substitutions, and insertions from a JSON file
        and return a new instance of ProbabilityDistributions.

        :param file_path: The path to the JSON file where the data is stored.
        :return: An instance of ProbabilityDistributions with the loaded data.
        """
        with open(file_path, "r") as f:
            data = json.load(f)

        # Create a new instance of the class without processing aligned_texts
        instance = cls.__new__(cls)

        # Initialize the counters (this is necessary to avoid uninitialized attributes)
        (
            instance.deletion_counts,
            instance.insertion_counts,
            instance.substitution_counts,
            instance.character_counts,
        ) = instance.initialize_counters()

        # Load the data into the instance
        instance.character_distribution = data["character_distribution"]
        instance.conditional = data["conditional"]
        instance.substitutions = data["substitutions"]
        instance.insertions = data["insertions"]

        # Add default values if they are not present in the loaded data
        if "default" not in instance.conditional:
            instance.add_default_values()

        return instance


class Character:
    def __init__(self, char):
        self.original = char
        self.current = char
        self.state = "Correct"
        self.insertions = []


class CharacterCorruptionEngine:
    def __init__(self, conditional_probs, substitution_table, insertion_table):
        self.conditional_probs = conditional_probs
        self.substitution_table = substitution_table
        self.insertion_table = insertion_table

        # Ensure default options exist
        if "default" not in self.conditional_probs:
            raise ValueError("conditional_probs must include a 'default' entry")
        if "default" not in self.substitution_table:
            raise ValueError("substitution_table must include a 'default' entry")
        if "default" not in self.insertion_table:
            raise ValueError("insertion_table must include a 'default' entry")

    def process_character(self, char):
        error_count = 0
        while True:
            if char.state == "Correct":
                char.state = self.choose_action(char.original)
                if char.state == "Correct":
                    return self.finalize(char), error_count
            elif char.state == "Substituted":
                char.current = self.substitute(char.original)
                error_count += 1  # Count as one substitution error
                if self.choose_action(char.current) != "Inserted":
                    return self.finalize(char), error_count
                char.state = "Inserted"
            elif char.state == "Deleted":
                error_count += 1  # Count as one deletion error
                return [], error_count
            elif char.state == "Inserted":
                inserted = self.insert_character(char.current)
                char.insertions.append(inserted)
                error_count += 1  # Count each insertion as one error
                if self.choose_action(inserted) != "Inserted":
                    return self.finalize(char), error_count
            char.current = char.insertions[-1] if char.insertions else char.current

    def choose_action(self, char):
        probs = self.conditional_probs.get(char, self.conditional_probs["default"])
        return random.choices(
            ["Correct", "Substituted", "Deleted", "Inserted"],
            weights=[
                probs["correct"],
                probs["substitute"],
                probs["delete"],
                probs["insert"],
            ],
        )[0]

    def substitute(self, char):
        sub_options = self.substitution_table.get(
            char, self.substitution_table["default"]
        )
        choices = list(sub_options.keys())
        weights = list(sub_options.values())
        return random.choices(choices, weights=weights)[0]

    def insert_character(self, prev_char):
        insert_options = self.insertion_table.get(
            prev_char, self.insertion_table["default"]
        )
        choices = list(insert_options.keys())
        weights = list(insert_options.values())
        return random.choices(choices, weights=weights)[0]

    def finalize(self, char):
        return [char.current] + char.insertions

    def corrupt_characters(self, text):
        corrupted_chars = []
        total_char_errors = 0
        total_chars = len(text)

        for char in text:
            original_char = char
            corrupted_char, error_count = self.process_character(Character(char))
            corrupted_chars.extend(corrupted_char)

            # Increment total character errors by the error count returned from process_character
            total_char_errors += error_count

        # Calculate CER
        corrupted_text = "".join(corrupted_chars)
        cer = total_char_errors / total_chars if total_chars > 0 else 0

        return corrupted_text, cer


class CorruptionEngine(CharacterCorruptionEngine):
    """
    Corrupts text based on a target WER and CER, and returns the corrupted text along with the actual WER, CER, and effective CER.
    """

    def __init__(
        self,
        conditional_probs,
        substitution_table,
        insertion_table,
        target_wer=1,
        target_cer=0.2,
    ):
        # Correctly initialize the parent class
        super().__init__(conditional_probs, substitution_table, insertion_table)

        self.target_wer = target_wer
        self.target_cer = target_cer

    def split_text(self, text):
        """
        Split the text into words, keeping punctuation and spaces as part of the words.
        This function ensures that spaces and punctuation are preserved in the corruption process.
        """
        return re.findall(r"\S+\s*", text)

    def corrupt_text(self, text):
        words = self.split_text(text)
        num_words = len([word for word in words if not word.isspace()])

        # Determine the number of words to corrupt based on the target WER
        num_words_to_corrupt = math.ceil(self.target_wer * num_words)
        words_to_corrupt_indices = random.sample(
            range(len(words)), num_words_to_corrupt
        )

        # Calculate the fraction of characters that will be corrupted
        selected_chars_count = sum(len(words[i]) for i in words_to_corrupt_indices)
        total_chars_count = len(text)
        selected_fraction = selected_chars_count / total_chars_count

        # Calculate the effective CER for the selected words and spaces
        effective_cer = self.target_cer / selected_fraction

        # Modify and renormalize probabilities based on the effective correct rate
        effective_correct_rate = 1 - effective_cer
        modified_conditional_probs = modify_and_renormalize_probs(
            self.conditional_probs,
            column="correct",
            desired_value=effective_correct_rate,
        )

        # Initialize a new corruption engine with modified probabilities
        modified_scrambler = CharacterCorruptionEngine(
            modified_conditional_probs, self.substitution_table, self.insertion_table
        )

        # Corrupt the selected words and track errors
        corrupted_words = []
        total_char_errors = 0
        chars_in_selected_words = 0  # Tracks the number of characters in selected words

        for i, word in enumerate(words):
            if i in words_to_corrupt_indices:
                corrupted_word, cer = modified_scrambler.corrupt_characters(word)
                total_char_errors += cer * len(word)  # Scale the CER by the word length
                chars_in_selected_words += len(
                    word
                )  # Count the characters in selected words
            else:
                corrupted_word = word  # Leave the word uncorrupted
            corrupted_words.append(corrupted_word)

        # Calculate the actual WER and CER
        actual_wer = len(words_to_corrupt_indices) / num_words if num_words > 0 else 0
        actual_cer = (
            total_char_errors / total_chars_count if total_chars_count > 0 else 0
        )
        effective_cer_value = (
            total_char_errors / chars_in_selected_words
            if chars_in_selected_words > 0
            else 0
        )

        # Join the corrupted words back into a single string
        corrupted_text = "".join(corrupted_words)

        return corrupted_text, actual_wer, actual_cer, effective_cer_value
