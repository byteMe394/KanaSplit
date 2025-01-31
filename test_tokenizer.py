import unittest
import requests
from unittest.mock import patch
from tokenizer import tokenize_text_with_pos, fetch_word_from_jisho


class TestJapaneseTokenizer(unittest.TestCase):

    def test_tokenize_text_with_pos(self):
        # Test input text
        text = "ハサミを買いたいんですが、文房具売り場は何回ですか？"

        # Expected tokenization output
        expected_tokens = [
            {"surface": "ハサミ", "pos": "名詞"},
            {"surface": "を", "pos": "助詞"},
            {"surface": "買い", "pos": "動詞"},
            {"surface": "たい", "pos": "助動詞"},
            {"surface": "ん", "pos": "名詞"},
            {"surface": "です", "pos": "助動詞"},
            {"surface": "が", "pos": "助詞"},
            {"surface": "、", "pos": "記号"},
            {"surface": "文房具", "pos": "名詞"},
            {"surface": "売り場", "pos": "名詞"},
            {"surface": "は", "pos": "助詞"},
            {"surface": "何", "pos": "名詞"},
            {"surface": "回", "pos": "名詞"},
            {"surface": "です", "pos": "助動詞"},
            {"surface": "か", "pos": "助詞"},
            {"surface": "？", "pos": "記号"},
        ]

        # Call the function
        tokens = tokenize_text_with_pos(text)
        print(tokens)

        # Assertions
        for expected, actual in zip(expected_tokens, tokens):
            self.assertEqual(expected["surface"], actual["surface"])
            self.assertEqual(expected["pos"], actual["pos"])

    @patch('tokenizer.requests.get')
    def test_fetch_word_from_jisho(self, mock_get):
        # Mock response for the Jisho API
        mock_response = {
            "data": [
                {
                    "slug": "ハサミ",
                    "japanese": [{"reading": "はさみ"}],
                    "senses": [
                        {"english_definitions": ["scissors", "shears"]}
                    ]
                }
            ]
        }

        # Configure the mock to return the mocked response
        mock_get.return_value.json.return_value = mock_response
        mock_get.return_value.status_code = 200

        # Call the function
        word_data = fetch_word_from_jisho("ハサミ")

        # Expected output
        expected_output = {
            "word": "ハサミ",
            "reading": "はさみ",
            "meanings": [["scissors", "shears"]]
        }

        # Assertions
        self.assertEqual(word_data, expected_output)

    @patch('tokenizer.requests.get')
    def test_fetch_word_from_jisho_failure(self, mock_get):
        # Configure the mock to simulate a request failure
        mock_get.side_effect = requests.exceptions.RequestException

        # Call the function
        word_data = fetch_word_from_jisho("nonexistentword")

        # Assertions
        self.assertIsNone(word_data)


if __name__ == '__main__':
    unittest.main()
