import re
import os
import shutil
from git import Repo
import json

from langchain_community.document_loaders.git import GitLoader

from langchain.text_splitter import Language
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Chroma





REPO_BASE = "test_repo"
DB_DIR = "db"


SUPPORTED_LANGUAGES =  {
    Language.PYTHON: {
        'suffixes': ['.py', '.pyx', '.pyi'],
        'language': Language.PYTHON
    },
    Language.JS: {
        'suffixes': ['.js', '.jsx', '.ts', '.tsx'],
        'language': Language.JS
    },
    Language.JAVA: {
        'suffixes': ['.java'],
        'language': Language.JAVA
    },
    Language.CPP: {
        'suffixes': ['.cpp', '.cc', '.cxx', '.h', '.hpp'],
        'language': Language.CPP
    },
    Language.CSHARP: {
        'suffixes': ['.cs'],
        'language': Language.CSHARP
    },
    Language.PHP: {
        'suffixes': ['.php'],
        'language': Language.PHP
    },
    Language.RUBY: {
        'suffixes': ['.rb'],
        'language': Language.RUBY
    },
    Language.RUST: {
        'suffixes': ['.rs'],
        'language': Language.RUST
    },
    Language.GO: {
        'suffixes': ['.go'],
        'language': Language.GO
    },
    Language.SCALA: {
        'suffixes': ['.scala'],
        'language': Language.SCALA
    },
    Language.KOTLIN: {
        'suffixes': ['.kt', '.kts'],
        'language': Language.KOTLIN
    },
    Language.LUA: {
        'suffixes': ['.lua'],
        'language': Language.LUA
    },
    Language.PERL: {
        'suffixes': ['.pl', '.pm'],
        'language': Language.PERL
    },
    Language.ELIXIR: {
        'suffixes': ['.ex', '.exs'],
        'language': Language.ELIXIR
    },
    Language.COBOL: {
        'suffixes': ['.cob', '.cbl'],
        'language': Language.COBOL
    }
}

TEXT_FILES = {
    '.txt', '.md', '.markdown', '.rst', '.adoc', '.asciidoc',
    '.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf',
    '.html', '.htm', '.css', '.scss', '.sass',
    '.csv', '.tsv', '.xml', '.rss', '.atom',
    '.mdx', '.log', '.readme', '.license', '.changelog',
    '.dockerfile', '.dockerignore', '.gitignore', '.gitattributes',
    '.env', '.env.example', '.env.local',
    '.sh', '.bash', '.zsh', '.fish',  # Shell scripts
    '.sql', '.psql', '.mysql',  # SQL files
    '.dockerfile', '.docker-compose.yml', '.docker-compose.yaml',
    '.yaml', '.yml',  # YAML files
    '.json', '.jsonc',  # JSON files
    '.xml', '.xsd', '.xslt',  # XML files
    '.html', '.htm', '.xhtml',  # HTML files
    '.css', '.scss', '.sass', '.less',  # CSS files
    '.md', '.markdown', '.mdown',  # Markdown files
    '.txt', '.text',  # Plain text files
    '.log', '.out', '.err',  # Log files
    '.ini', '.cfg', '.conf', '.config',  # Config files
    '.toml', '.lock',  # TOML files
    '.rss', '.atom', '.feed',  # Feed files
    '.csv', '.tsv', '.tab',  # Data files
    '.readme', '.license', '.changelog', '.contributing',  # Documentation
    '.gitignore', '.gitattributes', '.gitmodules',  # Git files
    '.dockerignore', '.dockerfile',  # Docker files
    '.env', '.env.example', '.env.local', '.env.production',  # Environment files
}


EMBEDDINGS_STATUS_FILE = "embeddings_status.json"

def load_embeddings_status():
    if os.path.exists(EMBEDDINGS_STATUS_FILE):
        with open(EMBEDDINGS_STATUS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_embeddings_status(status):
    with open(EMBEDDINGS_STATUS_FILE, "w") as f:
        json.dump(status, f)

def valid_github_url(url):
    print("Checking if github URL is valid")
    return re.match(r"^https://github.com/[\w\-]+/[\w\-]+/?$", url)

def clone_rep(url):
    print("Attempting to clone repo : ", url)
    repo_name = url.rstrip("/").split("/")[-1]
    repo_path = os.path.join(REPO_BASE, repo_name)
    print("Repo Name : ", repo_name)
    print("Repo Path : ", repo_path)

    # Clone if not already present
    if not os.path.exists(repo_path) or not os.listdir(repo_path):
        if os.path.exists(repo_path):
            shutil.rmtree(repo_path)
        Repo.clone_from(url, repo_path)
        print("Cloned repo successfully")
        return repo_path
    else:
        print("Repo already exists")
        return repo_path

def load_repo(repo_path):
    print("Loading repo")
    loader = GitLoader(
            repo_path=repo_path,
            # clone_url=url,
            branch="main",
            file_filter=create_language_filter()
        )
    documents = loader.load()
    return documents


def process_repo_documents(documents):
    print("Processing repo documents")
    processed_docs = []
    for doc in documents:
        file_ext = get_file_extension(doc.metadata.get('source', ''))

        language = None
        for lang, exts in SUPPORTED_LANGUAGES.items():
            if file_ext in exts:
                language = lang
                break
        try:
            if language:
                text_splitter = RecursiveCharacterTextSplitter.from_language(
                    language=language,
                    chunk_size=500,
                    chunk_overlap=20
                )
            else:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=20
                )
            chunks = text_splitter.split_documents([doc])
            processed_docs.extend(chunks)
        except Exception:
            default_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=20
            )
            processed_docs.extend(default_splitter.split_documents([doc]))
    return processed_docs

def computing_embeddings(processed_docs):
    print("Computing embeddings")
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(processed_docs, embedding=embeddings, persist_directory=DB_DIR)
    return vectordb

def load_vectordb():
    print("Loading vectordb from disk")
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    return vectordb


def get_file_extension(file_path):
    """Extract file extension from path"""
    return os.path.splitext(file_path)[1].lower()


def create_language_filter(languages=None):
    """Create a file filter for specified languages"""
    if languages is None:
        # Include all supported extensions
        allowed_extensions = []
        for value in SUPPORTED_LANGUAGES.values():
            allowed_extensions.extend(value["suffixes"])

        allowed_extensions.extend(TEXT_FILES)
    else:
        # Include only specified languages
        allowed_extensions = []
        for lang in languages:
            if lang in SUPPORTED_LANGUAGES:
                allowed_extensions.extend(SUPPORTED_LANGUAGES[lang]["suffixes"])
            elif lang == "TEXT":
                allowed_extensions.extend(TEXT_FILES)
    def file_filter(file_path):
        return get_file_extension(file_path) in allowed_extensions
    
    return file_filter