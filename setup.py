from setuptools import setup, find_packages

setup(
    name="ncode",
    version="1.0.0",
    description="A powerful local code generation system using LLaMA models",
    author="Ncode Team",
    packages=find_packages(),
    install_requires=[
        "llama-cpp-python>=0.2.0",
        "torch>=2.0.0",
        "rich>=13.0.0",
        "asyncio>=3.4.3",
        "pathlib>=1.0.1",
    ],
    entry_points={
        'console_scripts': [
            'ncode=backend.cli_client:main',
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Code Generators",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="code generation, llama, ai, development, mtp",
    project_urls={
        "Documentation": "https://github.com/yourusername/ncode/docs",
        "Source": "https://github.com/yourusername/ncode",
        "Issues": "https://github.com/yourusername/ncode/issues",
    }
)
