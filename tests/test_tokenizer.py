# test_tokenizer.py
import tempfile
import unittest
from tokenizer import Tokenizer

class TestTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(["Hello world", "Hello Python"])

    def test_fit_on_texts(self):
        expected_vocab_size = 4 # unique words: 'hello', 'world', 'python', '<OOV>'
        self.assertEqual(len(self.tokenizer.word_index), expected_vocab_size)

    def test_texts_to_sequences(self):
        sequences = self.tokenizer.texts_to_sequences(["Hello Python"])
        expected_sequences = [[self.tokenizer.word_index['hello'], self.tokenizer.word_index['python']]]
        self.assertEqual(sequences, expected_sequences)

    def test_sequences_to_texts(self):
        index_hello = self.tokenizer.word_index['hello']
        index_python = self.tokenizer.word_index['python']
        decoded_text = self.tokenizer.sequences_to_texts([[index_hello, index_python]])
        expected_text = ['hello python']
        self.assertEqual(decoded_text, expected_text)

    def test_custom_regex_hyphen(self):
        custom_tokenizer = Tokenizer(regex=r'\b[\w-]+\b')
        custom_tokenizer.fit_on_texts(["high-quality", "well-known", "text-processing"])
        self.assertIn('high-quality', custom_tokenizer.word_index)
        self.assertIn('well-known', custom_tokenizer.word_index)
        self.assertIn('text-processing', custom_tokenizer.word_index)

    def test_oov_token(self):
        self.tokenizer.fit_on_texts(["Hello world", "Hello Python"])
        sequences = self.tokenizer.texts_to_sequences(["Hello Java"])
        # "Java" is not in the training texts, so it should be replaced with the OOV token index
        expected_sequences = [[self.tokenizer.word_index['hello'], 1]]
        self.assertEqual(sequences, expected_sequences)

    def test_inverse_oov_token(self):
        self.tokenizer.fit_on_texts(["Hello world", "Hello Python"])
        # Sequence with OOV index (1)
        decoded_text = self.tokenizer.sequences_to_texts([[1, 1]])
        expected_text = [self.tokenizer.index_word[1] + ' ' + self.tokenizer.index_word[1]]
        self.assertEqual(decoded_text, expected_text)

    def test_default_and_custom_regex_commas_colons(self):
        # Using the default tokenizer
        default_tokenizer = Tokenizer()
        default_tokenizer.fit_on_texts(["example:word", "test,case", "standard:format,text"])

        # Check default tokenizer's behavior with commas and colons
        self.assertNotIn('example:word', default_tokenizer.word_index)
        self.assertNotIn('test,case', default_tokenizer.word_index)
        self.assertNotIn('standard:format,text', default_tokenizer.word_index)

        # Using a custom tokenizer that includes commas and colons
        custom_tokenizer = Tokenizer(regex=r'\b[\w,:]+\b')
        custom_tokenizer.fit_on_texts(["example:word", "test,case", "standard:format,text"])

        # Check custom tokenizer's behavior
        self.assertIn('example:word', custom_tokenizer.word_index)
        self.assertIn('test,case', custom_tokenizer.word_index)
        self.assertIn('standard:format,text', custom_tokenizer.word_index)

    def test_empty_and_blank_texts(self):
        sequences = self.tokenizer.texts_to_sequences(["", "    "])
        expected_sequences = [[], []]  # Empty sequences expected
        self.assertEqual(sequences, expected_sequences)

    def test_punctuation_handling(self):
        # Assuming default tokenizer splits on punctuation
        sequences = self.tokenizer.texts_to_sequences(["hello, world!"])
        self.assertNotIn(',', self.tokenizer.word_index)
        self.assertNotIn('!', self.tokenizer.word_index)

    def test_numeric_only_strings(self):
        sequences = self.tokenizer.texts_to_sequences(["123", "456"])
        # Assuming numbers are treated as OOV
        expected_sequences = [[1], [1]]
        self.assertEqual(sequences, expected_sequences)

    def test_case_sensitivity(self):
        # Since the tokenizer is case-insensitive, 'hello' and 'Hello' should have the same index
        sequences = self.tokenizer.texts_to_sequences(["hello", "Hello"])
        expected_sequence = [self.tokenizer.word_index['hello']]
        self.assertEqual(sequences, [expected_sequence, expected_sequence])

    def test_repetitive_and_overlapping_words(self):
        self.tokenizer.fit_on_texts(["hello hello", "hellohello"])
        sequences = self.tokenizer.texts_to_sequences(["hello hello", "hellohello"])
        expected_sequences = [
            [self.tokenizer.word_index['hello'], self.tokenizer.word_index['hello']],
            [self.tokenizer.word_index.get('hellohello', 1)]
        ]
        self.assertEqual(sequences, expected_sequences)

    def test_mixed_content(self):
        self.tokenizer.fit_on_texts(["abc123", "hello-world!"])

        # Tokenize the mixed content texts
        sequences = self.tokenizer.texts_to_sequences(["abc123", "hello-world!"])

        # Update expected sequences to reflect actual tokenizer behavior
        abc123_index = self.tokenizer.word_index.get('abc123', 1)  # Index of 'abc123'
        hello_index = self.tokenizer.word_index.get('hello', 1)    # Index of 'hello'
        world_index = self.tokenizer.word_index.get('world', 1)    # Index of 'world'

        expected_sequences = [[abc123_index], [hello_index, world_index]]

        self.assertEqual(sequences, expected_sequences)

    def test_save_load_tokenizer(self):
        with tempfile.NamedTemporaryFile() as temp_file:
            # Get the path of the temporary file
            temp_file_path = temp_file.name

            # Fit and save the tokenizer
            self.tokenizer.fit_on_texts(["Hello world", "Hello Python"])
            self.tokenizer.save(temp_file_path)

            # Load the tokenizer from the temporary file
            loaded_tokenizer = Tokenizer.load(temp_file_path)

            # Test that the loaded tokenizer is the same as the original
            self.assertEqual(self.tokenizer.word_index, loaded_tokenizer.word_index)
            self.assertEqual(self.tokenizer.index_word, loaded_tokenizer.index_word)

    def test_calculate_tfidf(self):
        # Assuming 'Hello' appears in both texts and 'world', 'Python' appear in one text each
        self.tokenizer.calculate_tfidf(["Hello world", "Hello Python"])

        # IDF for 'Hello' should be lower since it appears in both documents
        idf_hello = self.tokenizer.idf[self.tokenizer.word_index['hello']]
        idf_world = self.tokenizer.idf[self.tokenizer.word_index['world']]
        idf_python = self.tokenizer.idf[self.tokenizer.word_index['python']]

        self.assertTrue(idf_hello < idf_world and idf_hello < idf_python,
                        "IDF for 'hello' should be lower than 'world' and 'python'")

    def test_compound_word_normalization(self):
        # Test compound word normalization
        self.tokenizer.fit_on_texts(["counter-gambit plays", "counter gambit strategies", "countergambit moves"])
        normalized_index = self.tokenizer.word_index.get('countergambit')

        # Check if different variations are normalized to 'countergambit'
        sequences = self.tokenizer.texts_to_sequences(["counter-gambit", "counter gambit", "countergambit"])
        expected_sequences = [[normalized_index], [normalized_index], [normalized_index]]

        self.assertEqual(sequences, expected_sequences, "Compound word variations should be normalized to the same form")

    def test_camel_case_splitting_and_tokenization(self):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(["CaroKann Defense", "FriedLiver Attack", "King's Gambit"])

        sequences = tokenizer.texts_to_sequences(["CaroKann", "FriedLiver", "King's"])
        expected_sequences = [
            [tokenizer.word_index.get('caro', 1), tokenizer.word_index.get('kann', 1)],  # 'CaroKann' -> 'caro', 'kann'
            [tokenizer.word_index.get('fried', 1), tokenizer.word_index.get('liver', 1)],  # 'FriedLiver' -> 'fried', 'liver'
            [tokenizer.word_index.get("king", 1)]  # "King's" -> 'king', 's' is filtered out
        ]
        self.assertEqual(sequences, expected_sequences, "Camel case words should be split and tokenized correctly")

class TestUseMetaphone(unittest.TestCase):

    def test_metaphone_usage(self):
        # Test that metaphone is used when use_metaphone is True
        tokenizer = Tokenizer(use_metaphone=True)
        input_text = "Example"
        expected_output = ['AKSMPL']
        self.assertEqual(tokenizer.tokenize(input_text), expected_output)

    def test_metaphone_with_multiple_words(self):
        # Test metaphone with multiple words
        tokenizer = Tokenizer(use_metaphone=True)
        input_text = "Multiple words OrthoSchnapp"
        expected_output = ['MLTPL', 'ARTS', 'FRTS', 'AR0XNP', 'ARTXNP']
        self.assertEqual(tokenizer.tokenize(input_text), expected_output)

    def test_empty_string(self):
        # Test behavior with an empty string
        tokenizer = Tokenizer(use_metaphone=True)
        input_text = ""
        expected_output = []
        self.assertEqual(tokenizer.tokenize(input_text), expected_output)

    def test_metaphone_with_compound_word(self):
        # Test metaphone with compound word processing
        tokenizer = Tokenizer(use_metaphone=True)
        input_text = "Counter gambit"
        expected_output = ['KNTRKMPT']
        self.assertEqual(tokenizer.tokenize(input_text), expected_output)


if __name__ == '__main__':
    unittest.main()
