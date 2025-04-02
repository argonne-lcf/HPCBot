from setuptools import setup, find_packages

setup(
    name='hpcbot',  
    version='0.1.0',  
    description='HPCBot: A CLI tool for document-based QA and generation.',  
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/btrungvo/hpcbot',
    author='Trung Vo',
    author_email='trungvo.usth@gmail.com',
    license='MIT',  
    packages=find_packages(),
    install_requires=[
        'langchain_community',
        "langchain-text-splitters",
        'openai',
        "transformers",
        "argparse",
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    entry_points={
        "console_scripts": [
            "hpcbot=hpcbot.cli:main",
        ],
    },
    python_requires='>=3.6',
)