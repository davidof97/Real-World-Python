"""
Compare writing styles with simple stylometry tests.

Expected files in the same folder:
- hound.txt -> Arthur Conan Doyle, The Hound of the Baskervilles
- war.txt   -> H. G. Wells, The War of the Worlds
- lost.txt  -> unknown text, The Lost World
"""
from pathlib import Path
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

# Line styles used in plots so the charts are readable even without color.
LINES = ['-', ':', '--']

# Author Labels mapped to corpus file names.
CORPUS_FILES = {
    "doyle" : "hound.txt",
    "wells": "war.txt",
    "unknown" : "lost.txt"
}

def ensure_nltk_data() -> None:
    """Download required NLTK data packages if they are missing."""
    required_resources = [
        ("punkt", "tokenizers/punkt"),
        ("punkt_tab", "tokenizers/punkt_tab"),
        ("stopwords", "corpora/stopwords"),
        ("averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger"),
        ("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng")
    ]
    
    for package_name, resource_path in required_resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(package_name, quiet=True)

def text_to_string(filename: str) -> str:
    """Read a text file and return its contents as one string"""
    path = Path(filename)
    
    if not path.exists():
        raise FileNotFoundError(f"Couldn't find {filename!r}. Put this file in the same folder as stylometry.py.")
    
    return path.read_text(encoding="utf-8", errors="ignore")

def make_word_dict(strings_by_author: dict[str, str]) -> dict[str, list[str]]:
    """Tokenize each author's text and return lowercase alphabetic words."""
    words_by_author = {}
    
    for author, text in strings_by_author.items():
        tokens = nltk.word_tokenize(text)
        words_by_author[author] = [token.lower() for token in tokens if token.isalpha()]
    
    return words_by_author

def find_shortest_corpus(words_by_author: dict[str, list[str]]) -> int:
    """Return the word count of the shortest corpus."""
    word_counts = []
    
    for author, words in words_by_author.items():
        word_count = len(words)
        word_counts.append(word_count)
        print(f"Word count for {author}: {word_count}")
    
    shortest = min(word_counts)
    print(f"Shortest corpus length: {shortest}\n")
    return shortest

def word_length_test(words_by_author: dict[str, list[str]], len_shortest_corpus: int) -> None:
    """Plot how often each author uses words of different lengths."""
    plt.figure(1)
    
    for i, (author, words) in enumerate(words_by_author.items()):
        trimmed_words = words[:len_shortest_corpus]
        word_lengths = [len(word) for word in trimmed_words]
        frequency = nltk.FreqDist(word_lengths)
        
        frequency.plot(15, linestyle=LINES[i], label=author, title="Word length frequency")
    
    plt.legend()
    plt.ylabel("Count")
    plt.xlabel("Word length")
    
def stopwords_test(words_by_author: dict[str, list[str]], len_shortest_corpus: int) -> None:
    """Plot the most common stopwords used by each author."""
    plt.figure(2)
    stop_words = set(stopwords.words("english"))
    
    for i, (author, words) in enumerate(words_by_author.items()):
        trimmed_words = words[:len_shortest_corpus]
        author_stopwords = [word for word in trimmed_words if word in stop_words]
        frequency = nltk.FreqDist(author_stopwords)
        
        frequency.plot(50, linestyle=LINES[i], label=author, title="50 most common stopwords")
        
        plt.legend()
        plt.ylabel("Count")
        plt.xlabel("Stopword")

def parts_of_speech_test(words_by_author: dict[str, list[str]], len_shortest_corpus: int) -> None:
    """Plot the most common parts of speech used by each author."""
    plt.figure(3)
    
    for i, (author, words) in enumerate(words_by_author.items()):
        trimmed_words = words[:len_shortest_corpus]
        pos_tags = [tag for _, tag in nltk.pos_tag(trimmed_words)]
        frequency = nltk.FreqDist(pos_tags)
        
        frequency.plot(35, linestyle=LINES[i], label=author, title="Part-of-speech frequency.")
    
    plt.legend()
    plt.ylabel("Count")
    plt.xlabel("Part of speech")

def vocab_test(words_by_author: dict[str, list[str]]) -> str:
    """Compare vocabulary with a chi-square statistic."""
    chisquared_by_author = {}
    unknown_words = words_by_author["unknown"]
    
    for author, author_words in words_by_author.items():
        if author == "unknown":
            continue
        
        combined_corpus = author_words + unknown_words
        author_proportion = len(author_words) / len(combined_corpus)
        combined_frequency = nltk.FreqDist(combined_corpus)
        author_frequency = nltk.FreqDist(author_words)
        most_common_words = combined_frequency.most_common(1000)
        
        chisquared = 0.0
        
        for word, combined_count in most_common_words:
            observed_count = author_frequency[word]
            expected_count = combined_count * author_proportion
            chisquared += ((observed_count - expected_count) ** 2) / expected_count
        
        chisquared_by_author[author] = chisquared
        print(f"Chi-square for {author}: {chisquared:.1f}")
        
    most_likely_author = min(chisquared_by_author, key=chisquared_by_author.get)
    print(f"Vocabulary points to: {most_likely_author}\n")
    return most_likely_author

def jaccard_test(words_by_author: dict[str, list[str]], len_shortest_corpus: int) -> str:
    """Compare unique-word overlap with the Jaccard similarity index."""
    jaccard_by_author = {}
    unknown_words = set(words_by_author["unknown"][:len_shortest_corpus])
    
    for author, words in words_by_author.items():
        if author == "unknown":
            continue
        
        author_words = set(words[:len_shortest_corpus])
        shared_words = author_words.intersection(unknown_words)
        all_unique_words = author_words.union(unknown_words)
        jaccard_similarity = len(shared_words) / len(all_unique_words)
        
        jaccard_by_author[author] = jaccard_similarity
        print(f"Jaccard index for {author}: {jaccard_similarity:.4f}")
    
    most_likely_author = max(jaccard_by_author, key=jaccard_by_author.get)
    print(f"Jaccard similarity points to: {most_likely_author}")
    return most_likely_author

def main() -> None:
    """Run all stylometry tests."""
    ensure_nltk_data()
    
    strings_by_author = {
        author: text_to_string(filename) for author, filename in CORPUS_FILES.items()
    }
    
    print("Preview of Doyle corpus:")
    print(strings_by_author["doyle"][:300], "\n")
    
    words_by_author = make_word_dict(strings_by_author)
    len_shortest_corpus = find_shortest_corpus(words_by_author)
    
    word_length_test(words_by_author, len_shortest_corpus)
    stopwords_test(words_by_author, len_shortest_corpus)
    parts_of_speech_test(words_by_author, len_shortest_corpus)
    
    vocab_winner = vocab_test(words_by_author)
    jaccard_winner = jaccard_test(words_by_author, len_shortest_corpus)
    
    print("\nFinal hints:")
    print(f"- Vocabulary test winner: {vocab_winner}")
    print(f"- Jaccard test winner: {jaccard_winner}")
    print("- Check the plots to compare word length, stopwords and POS usage.")
    
    plt.show(block=True)

if __name__ == "__main__":
    main()