# KanaSplit - Japanese Text Tokenizer

![KanaSplit Logo](assets/dancing_shigure.gif)  
*A powerful and efficient Japanese text tokenizer with POS tagging and Jisho.org integration.*

## 📌 Overview
KanaSplit is a Japanese text tokenizer designed to break down Japanese sentences into meaningful tokens while providing part-of-speech (POS) tagging. It integrates with [Jisho.org](https://jisho.org) to fetch additional lexical data for individual words. The tool is built using **MeCab**, a popular morphological analyzer for the Japanese language.

## 🚀 Features
- 🔹 **Tokenization**: Splits Japanese sentences into words and morphemes.
- 🔹 **POS Tagging**: Provides grammatical category for each token.
- 🔹 **Furigana Support**: Extracts readings for kanji words.
- 🔹 **Jisho.org API Integration**: Retrieves word meanings and definitions.
- 🔹 **Command-Line Interface (CLI)**: Allows easy text tokenization from the terminal.
- 🔹 **Error Logging**: Captures and logs API failures for debugging.

## 🔧 Installation
KanaSplit requires **Python 3.6+** and **MeCab** to be installed.

### **1️⃣ Install Dependencies**
```sh
pip install -r requirements.txt
```

### **2️⃣ Install MeCab (if not installed)**
#### **For Windows**:
```sh
choco install mecab mecab-ipadic
```
#### **For macOS**:
```sh
brew install mecab mecab-ipadic
```
#### **For Linux (Ubuntu/Debian)**:
```sh
sudo apt install mecab mecab-ipadic
```

### **3️⃣ Install KanaSplit**
```sh
pip install .
```

## 📖 Usage
### **1️⃣ Tokenize Japanese Text**
```sh
kanasplit-cli "私はお寿司を食べたいです。"
```
**Output:**
```
- 私 (名詞)
- は (助詞)
- お (接頭詞)
- 寿司 (名詞)
- を (助詞)
- 食べ (動詞)
- たい (助動詞)
- です (助動詞)
- 。 (記号)
```

### **2️⃣ Fetch Word Definitions from Jisho.org**
```sh
kanasplit-cli -w "寿司"
```
**Output:**
```
Word: 寿司
Reading: すし
Meanings: ['sushi', 'range of dishes made with vinegared rice combined with fish, vegetables, egg, etc.']
```

## 🛠 API Usage
You can also use KanaSplit in your Python scripts:
```python
from tokenizer import tokenize_text_with_pos, fetch_word_from_jisho

text = "ハサミを買いたいんですが、文房具売り場は何回ですか？"
tokens = tokenize_text_with_pos(text)
print(tokens)

word_data = fetch_word_from_jisho("寿司")
print(word_data)
```

## 📝 Configuration
KanaSplit logs errors in `errors.log`. You can configure logging settings in `tokenizer.py`.

## 🏗️ Development & Contribution
Want to contribute? Feel free to fork this repository and submit a pull request!
```sh
git clone https://github.com/byteMe394/KanaSplit.git
cd KanaSplit
```

## 📜 License
This project is licensed under the MIT License.

## 👨‍💻 Author
**José Trujillo**  
[GitHub](https://github.com/byteMe394) | [Email](mailto:joseantonio_tf@outlook.com)
